import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Create directory for figures if it doesn't exist
os.makedirs('figs', exist_ok=True)

# Load the speedup ratios from the JSON file
with open('speedup_ratios.json', 'r') as f:
    speedup_results = json.load(f)

# Extract unique values for num_speculative_tokens and ngram_prompt_lookup_max
spec_tokens = sorted(set(item['num_speculative_tokens'] for item in speedup_results))
ngram_lookup = sorted(set(item['ngram_prompt_lookup_max'] for item in speedup_results))

# Create a dictionary to store speedup data for each combination
# We'll create separate dataframes for each batch size
batch_sizes = sorted(set(item['batch_size'] for item in speedup_results))
batch_dfs = {}

for batch in batch_sizes:
    # Create a DataFrame for this batch size
    df = pd.DataFrame(index=spec_tokens, columns=ngram_lookup)
    
    for s in spec_tokens:
        for n in ngram_lookup:
            # Filter results for this combination and batch size
            filtered = [item for item in speedup_results 
                       if item['num_speculative_tokens'] == s and 
                          item['ngram_prompt_lookup_max'] == n and
                          item['batch_size'] == batch]
            
            # Set speedup ratio if there's a result
            if filtered:
                df.loc[s, n] = filtered[0]['speedup_ratio']
    
    batch_dfs[batch] = df

# Choose batch size 8 for the main visualization (typically best performance)
# Change this to other batch sizes if needed
batch_size_to_plot = 1
df_to_plot = batch_dfs[batch_size_to_plot]
df_to_plot = df_to_plot.astype(float)
print(df_to_plot)
# Create the heatmap with larger font sizes and compact layout
plt.figure(figsize=(10, 8))

# Set font sizes
plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16

# Create the heatmap
ax = sns.heatmap(df_to_plot, 
                annot=True,          # Show values in cells
                fmt=".2f",   # 截断到2位小数而非四舍五入
                cmap="viridis",      # Color map
                linewidths=0.5,      # Grid line width
                annot_kws={"size": 12},  # Size of numbers in cells
                cbar_kws={'label': 'Speedup Ratio'})

# Set title and labels
plt.title(f'Speedup Ratio (Batch Size = {batch_size_to_plot})', fontsize=18, pad=20)
plt.xlabel('N-gram Prompt Lookup Max', fontsize=16, labelpad=10)
plt.ylabel('Number of Speculative Tokens', fontsize=16, labelpad=10)

# Increase tick label size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Adjust colorbar font size
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Speedup Ratio', fontsize=16)

# Make the plot more compact
plt.tight_layout()

# Save the figure
plt.savefig(f'figs/speedup_heatmap_compact_{batch_size_to_plot}.png', dpi=300, bbox_inches='tight')
print(f"Compact heatmap saved as 'figs/speedup_heatmap_compact.png'")

# Find the best configuration for displaying
best_config = None
max_speedup = 0

for s in spec_tokens:
    for n in ngram_lookup:
        if df_to_plot.loc[s, n] > max_speedup:
            max_speedup = df_to_plot.loc[s, n]
            best_config = (s, n)

if best_config:
    print(f"\nBest configuration for batch size {batch_size_to_plot}:")
    print(f"Speculative tokens: {best_config[0]}")
    print(f"N-gram lookup max: {best_config[1]}")
    print(f"Speedup ratio: {max_speedup:.2f}x") 