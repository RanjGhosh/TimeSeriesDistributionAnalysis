import matplotlib.pyplot as plt
import seaborn as sns

# Data (original slopes in N/s)
slopes_40 = [0.0007, 0.0010, 0.0009, 0.0009, 0.0010]
slopes_80 = [0.0005, 0.0006, 0.0010, 0.0008, 0.0007]

# Convert slope to N/mm using strain rate (10 mm/min = 10/60 mm/s)
strain_rate = 10 / 60  # mm/s
slopes_40_converted = [s / strain_rate for s in slopes_40]
slopes_80_converted = [s / strain_rate for s in slopes_80]

# Prepare data
data = slopes_40_converted + slopes_80_converted
labels = ['40 wires'] * len(slopes_40_converted) + ['80 wires'] * len(slopes_80_converted)

# Plot violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(x=labels, y=data, palette=['red', 'blue'], inner='box', linewidth=1)

plt.title("Violin Plot of Stiffness (Force/Displacement)")
plt.ylabel("Stiffness (10^-3 N/mm)")
plt.grid(True, linestyle='--', alpha=0.4)

# Save and show
plt.tight_layout()
plt.savefig("stiffness_violin_plot.png")
plt.show()
print("🎻 Violin plot saved as stiffness_violin_plot.png")
