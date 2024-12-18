import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
import evaluate
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
import uuid
import numpy as np

NUM_PREPROCESSING_WORKERS = 2


# creating debiasing trainer
import torch
class DataCartographyTrainer(Trainer):
    def __init__(self, *args, processing_class, **kwargs):
        super().__init__(*args, **kwargs)
        self.processing_class = processing_class
        self.example_probs = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")

        mask = (
            (input_ids != self.processing_class.cls_token_id) & 
            (input_ids != self.processing_class.pad_token_id) & 
            (input_ids != self.processing_class.sep_token_id)
        )

        decoded_texts = []
        for batch_input_ids, batch_mask in zip(input_ids, mask):
            cleaned_input_ids = batch_input_ids[batch_mask]
            full_text = self.processing_class.decode(cleaned_input_ids)
            
            parts = full_text.split()
            decoded_texts.append({
                'premise': ' '.join(parts[:len(parts)//2]),
                'hypothesis': ' '.join(parts[len(parts)//2:])
            })

        outputs = model(**inputs)
        logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)
        correct_probs = probs[range(len(labels)), labels]

        for idx, (text_info, prob) in enumerate(zip(decoded_texts, correct_probs)):
            if 'full_text' in text_info:
                key = text_info['full_text']
            else:
                key = (text_info['premise'], text_info['hypothesis'])
            if key not in self.example_probs:
                self.example_probs[key] = []
            
            self.example_probs[key].append(prob.item())

        loss = outputs.loss
        if loss is None:
            raise ValueError("Model returned None for the loss. Check your model's forward pass.")

        return (loss, outputs) if return_outputs else loss

    def create_statistics(self):
        example_stats = {
            f"{premise}:{hypothesis}": {
                "mean": np.mean(probs),
                "std": np.std(probs)
            }
            for (premise, hypothesis), probs in self.example_probs.items()
        }

        with open("fp-dataset-artifacts/data_mapping/data_statistics.json", "w") as f:
            import json
            json.dump(example_stats, f, indent=4)


from transformers import DataCollatorWithPadding

class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        batch['labels'] = torch.tensor([f['labels'] for f in features])
        batch['example_ids'] = torch.tensor([f['example_ids'] for f in features])
        
        return batch



def main():
    argp = HfArgumentParser(TrainingArguments)
    task = 'nli'
    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    training_args, args = argp.parse_args_into_dataclasses()


    dataset_id = 'snli'
    eval_split = 'validation'
    dataset = datasets.load_dataset(dataset_id)
    task_kwargs = {'num_labels': 3}


    model_class = AutoModelForSequenceClassification
    model = model_class.from_pretrained(args.model, **task_kwargs)
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)


    prepare_train_dataset = prepare_eval_dataset = \
        lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None

    with open('fp-dataset-artifacts/data_mapping/data_statistics.json', 'r') as file:
        data = json.load(file)

    filtered_pairs = set()
    for key, value in data.items():
        if value['mean'] > 0.5 and value['std'] < 0.2:
            filtered_pairs.add(key)

    if training_args.do_train:
        train_dataset = dataset['train']
        print(type(train_dataset))
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        
        new_dataset = []
        for i in range(len(train_dataset)):
            cur_example = train_dataset[i]
            cur_key = cur_example["premise"] + ":" + cur_example["hypothesis"]
            if cur_key not in filtered_pairs:
                new_dataset.append(cur_example)  

        from datasets import Dataset
        train_dataset = Dataset.from_list(new_dataset)          
            
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Select the training configuration
    eval_kwargs = {}
    compute_metrics = compute_accuracy
    

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = DataCartographyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions,
    )
    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)
        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')

    # save statistics
    trainer.create_statistics()


if __name__ == "__main__":
    main()