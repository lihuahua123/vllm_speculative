import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['DASpec', 'Smart', 'Deep', 'Nospec']
qps_15 = [4307.15, 2658.42, 4118.47, 4318.01]
qps_20 = [4630.59, 2771.30, 4401.52, 4626.05]
qps_25 = [4837.16, 2814.14, 4507.82, 4850.93]

# Set the width of the bars
barWidth = 0.2

# Set the positions of the bars on X axis
r1 = np.arange(len(methods))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Create the figure and axis
plt.figure(figsize=(12, 6))

# Create the bars
plt.bar(r1, qps_15, width=barWidth, label='QPS 15', color='skyblue')
plt.bar(r2, qps_20, width=barWidth, label='QPS 20', color='lightgreen')
plt.bar(r3, qps_25, width=barWidth, label='QPS 25', color='salmon')

# Add labels and title
plt.xlabel('Methods')
plt.ylabel('QPS')
plt.title('QPS Comparison Across Different Methods')
plt.xticks([r + barWidth for r in range(len(methods))], methods)

# Add a legend
plt.legend()

# Add grid for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('qps_comparison.png')
plt.close() 