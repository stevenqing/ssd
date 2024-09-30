import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 2, 9)  # 0 to 2 in 9 steps
algorithms = ['Simul-Ind', 'Simul-Co', 'SVO', 'CGA', 'SL', 'AgA (α = 100)']
data = {
    'Simul-Ind': 50 + 50 * np.sin(x * np.pi),
    'Simul-Co': 75 + 25 * np.sin(x * np.pi),
    'SVO': 100 + 20 * np.cos(x * np.pi),
    'CGA': 25 + 75 * np.sin(x * np.pi),
    'SL': -25 + 50 * np.cos(x * np.pi),
    'AgA (α = 100)': 125 + 25 * np.sin(x * np.pi)
}

# Set up the plot
plt.figure(figsize=(10, 6))
plt.title('Social Welfare vs Total Steps', fontsize=16)
plt.xlabel('Total Steps (1e7)', fontsize=12)
plt.ylabel('Social Welfare', fontsize=12)

# Plot each algorithm
for algo in algorithms:
    plt.plot(x, data[algo], label=algo, linewidth=2)

# Customize the plot
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.xlim(0, 2)
plt.ylim(-100, 150)

# Set x-axis ticks
plt.xticks(np.arange(0, 2.25, 0.25))

# Add a horizontal line at y=0
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()