import json

original_result = []
contrast_result = []

with open("fp-dataset-artifacts/original_output/eval_predictions.jsonl", "r") as file:
    for line in file:
        data = json.loads(line.strip())
        inner_array = [
            data["premise"],
            data["hypothesis"],
            data["label"],
            data["predicted_label"],
            data["label"] == data["predicted_label"]
        ]
        original_result.append(inner_array)

with open("fp-dataset-artifacts/contrast_output/eval_predictions.jsonl", "r") as file:
    for line in file:
        data = json.loads(line.strip())
        inner_array = [
            data["premise"],
            data["hypothesis"],
            data["label"],
            data["predicted_label"],
            data["label"] == data["predicted_label"]
        ]
        contrast_result.append(inner_array)

result = []

for i in range(len(original_result)):
    cur_original = original_result[i]
    cur_contrast = contrast_result[i]

    if cur_original[4] != cur_contrast[4]:
        temp = []
        for i in range(4):
            temp.append(cur_original[i])
        for i in range(4):
            temp.append(cur_contrast[i])

        result.append(temp)

for i in range(len(result)):
    cur_example = result[i]
    print("OP: " + cur_example[0] + " " + "OH: " + cur_example[1] + " " + "OL: " + str(cur_example[2]) + " " + "OPL: " + str(cur_example[3]))
    print("CP: " + cur_example[4] + " " + "CH: " + cur_example[5] + " " + "CL: " + str(cur_example[6]) + " " + "CPL: " + str(cur_example[7]))
    print()