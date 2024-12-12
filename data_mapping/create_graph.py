import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Load the data from the JSON file
with open('data_statistics.json', 'r') as file:
    data = json.load(file)

# Extract mean and std values
means = np.array([item['mean'] for item in data.values()])
stds = np.array([item['std'] for item in data.values()])

# Create a custom colormap
colors = ['blue', 'royalblue', 'cornflowerblue', 'lightskyblue', 'aquamarine', 
          'yellow', 'orange', 'salmon', 'tomato', 'red']
n_bins = 10  # 10 color bins for 0.1 intervals
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Create the scatter plot
plt.figure(figsize=(12, 8))
import json
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the JSON file
with open('data_statistics.json', 'r') as file:
    data = json.load(file)

# Extract mean and std values
means = np.array([item['mean'] for item in data.values()])
stds = np.array([item['std'] for item in data.values()])

# Define classification thresholds
std_threshold = .2
confidence_threshold = 0.5  # Confidence threshold

# Classify points
ambiguous = stds > std_threshold
hard = (means < confidence_threshold) & (stds <= std_threshold)
easy = (means >= confidence_threshold) & (stds <= std_threshold)

# Create the scatter plot
plt.figure(figsize=(14, 10))

# Plot each category
plt.scatter(stds[easy], means[easy], color='red', alpha=0.6, s=10, label='Easy-to-learn')
plt.scatter(stds[hard], means[hard], color='blue', alpha=0.6, s=10, label='Hard-to-learn')
plt.scatter(stds[ambiguous], means[ambiguous], color='purple', alpha=0.6, s=10, label='Ambiguous')

plt.xlabel('Variability (std)')
plt.ylabel('Confidence (mean)')
plt.title('Confidence vs Variability')

# Add labels above each section
plt.text(0.1, 0.9, 'Easy-to-learn', color='black', fontsize=24, transform=plt.gca().transAxes)
plt.text(0.1, 0.1, 'Hard-to-learn', color='black', fontsize=24, transform=plt.gca().transAxes)
plt.text(0.8, 0.5, 'Ambiguous', color='black', fontsize=24, transform=plt.gca().transAxes)

# Add a grid for better readability
plt.grid(True, linestyle=':', alpha=0.3)

# Save the plot to a file
plt.savefig('classified_confidence_vs_variability_with_labels.png', dpi=300)
