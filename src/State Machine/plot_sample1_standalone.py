import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from importlib import util

# Import from the original file
spec = util.spec_from_file_location("numpy_trial", os.path.join(os.path.dirname(__file__), "Numpy trial.py"))
numpy_trial = util.module_from_spec(spec)
spec.loader.exec_module(numpy_trial)

# Use the imported Sample1 array (already calculated in the original file)
Sample1 = numpy_trial.Sample1

# Create comprehensive visualization with one plot per row
fig = plt.figure(figsize=(16, 14))

# Create a subplot for each row
num_rows = Sample1.shape[0]
num_cols = 2
num_rows_grid = (num_rows + num_cols - 1) // num_cols  # Calculate grid rows needed
for i in range(num_rows):
    ax = plt.subplot(num_rows_grid, num_cols, i + 1)
    ax.plot(Sample1[i], linewidth=1.5, color='steelblue')
    ax.fill_between(range(Sample1.shape[1]), Sample1[i], alpha=0.3, color='steelblue')
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Value')
    ax.set_title(f'Row {i} - Data Visualization')
    ax.grid(True, alpha=0.3)
    
    # Add statistics to each subplot
    mean_val = np.mean(Sample1[i])
    std_val = np.std(Sample1[i])
    ax.axhline(mean_val, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Mean: {mean_val:.2f}')
    ax.axhline(mean_val + std_val, color='orange', linestyle=':', linewidth=1, alpha=0.7, label=f'Std: ±{std_val:.2f}')
    ax.axhline(mean_val - std_val, color='orange', linestyle=':', linewidth=1, alpha=0.7)
    ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()
