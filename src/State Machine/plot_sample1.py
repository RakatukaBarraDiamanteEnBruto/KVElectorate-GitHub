import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from importlib import util

# Import from the original file (has space in name)
spec = util.spec_from_file_location("numpy_trial", os.path.join(os.path.dirname(__file__), "Numpy trial.py"))
numpy_trial = util.module_from_spec(spec)
spec.loader.exec_module(numpy_trial)
Sample1 = numpy_trial.Sample1
fig = plt.figure(figsize=(16, 12))

# Subplot 1: Heatmap of the entire 2D array
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.imshow(Sample1, aspect='auto', cmap='viridis', interpolation='nearest')
ax1.set_xlabel('Column Index')
ax1.set_ylabel('Row Index')
ax1.set_title('Sample1 Array - Heatmap')
plt.colorbar(im1, ax=ax1, label='Magnitude')

# Subplot 2: Line plot showing each row
ax2 = plt.subplot(2, 2, 2)
for i in range(Sample1.shape[0]):
    ax2.plot(Sample1[i], label=f'Row {i}', linewidth=1)
ax2.set_xlabel('Column Index')
ax2.set_ylabel('Value')
ax2.set_title('Sample1 Array - Individual Rows')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: 3D surface plot representation (stacked areas)
ax3 = plt.subplot(2, 2, 3)
for i in range(Sample1.shape[0]):
    ax3.fill_between(range(Sample1.shape[1]), Sample1[i] + i*500, i*500, alpha=0.6, label=f'Row {i}')
ax3.set_xlabel('Column Index')
ax3.set_ylabel('Value (stacked)')
ax3.set_title('Sample1 Array - Stacked Area Chart')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Statistics summary as a bar chart
ax4 = plt.subplot(2, 2, 4)
row_means = np.mean(Sample1, axis=1)
row_stds = np.std(Sample1, axis=1)
x_pos = np.arange(Sample1.shape[0])
ax4.bar(x_pos, row_means, yerr=row_stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
ax4.set_xlabel('Row Index')
ax4.set_ylabel('Mean Value')
ax4.set_title('Sample1 Array - Row Statistics (Mean ± Std Dev)')
ax4.set_xticks(x_pos)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
