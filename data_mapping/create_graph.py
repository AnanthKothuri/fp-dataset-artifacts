import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

with open('data_statistics.json', 'r') as file:
    data = json.load(file)

means = np.array([item['mean'] for item in data.values()])
stds = np.array([item['std'] for item in data.values()])

colors = ['blue', 'royalblue', 'cornflowerblue', 'lightskyblue', 'aquamarine', 
          'yellow', 'orange', 'salmon', 'tomato', 'red']
n_bins = 10
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)



plt.figure(figsize=(12, 8))
import json
import matplotlib.pyplot as plt
import numpy as np

with open('data_statistics.json', 'r') as file:
    data = json.load(file)

means = np.array([item['mean'] for item in data.values()])
stds = np.array([item['std'] for item in data.values()])


std_threshold = .2
confidence_threshold = 0.5
ambiguous = stds > std_threshold
hard = (means < confidence_threshold) & (stds <= std_threshold)
easy = (means >= confidence_threshold) & (stds <= std_threshold)

plt.figure(figsize=(14, 10))
plt.scatter(stds[easy], means[easy], color='red', alpha=0.6, s=10, label='Easy-to-learn')
plt.scatter(stds[hard], means[hard], color='blue', alpha=0.6, s=10, label='Hard-to-learn')
plt.scatter(stds[ambiguous], means[ambiguous], color='purple', alpha=0.6, s=10, label='Ambiguous')

plt.xlabel('Variability (std)')
plt.ylabel('Confidence (mean)')
plt.title('Confidence vs Variability')
plt.text(0.1, 0.9, 'Easy-to-learn', color='black', fontsize=24, transform=plt.gca().transAxes)
plt.text(0.1, 0.1, 'Hard-to-learn', color='black', fontsize=24, transform=plt.gca().transAxes)
plt.text(0.8, 0.5, 'Ambiguous', color='black', fontsize=24, transform=plt.gca().transAxes)
plt.grid(True, linestyle=':', alpha=0.3)
plt.savefig('classified_confidence_vs_variability_with_labels.png', dpi=300)
