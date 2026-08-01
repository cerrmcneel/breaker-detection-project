import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set premium aesthetic
plt.style.use('dark_background')
sns.set_context("talk")
accent_color = "#00f2fe"  # PanelSafe Cyan
error_color = "#ff4b2b"   # PanelSafe Red
text_color = "#ffffff"

# 1. Create the Dataset
data = {
    'Environment': ['Synthetic (Grammar)', 'Field (Real World)', 'Field (HITL + Context)'],
    'mAP@50': [0.974, 0.420, 0.920],
    'Reliability': ['High', 'Low', 'Production-Ready']
}
df = pd.DataFrame(data)

# 2. Initialize Plot
plt.figure(figsize=(12, 7))
ax = sns.barplot(
    x='Environment', 
    y='mAP@50', 
    data=df, 
    palette=[accent_color, error_color, "#2ecc71"], # Cyan, Red, Green
    alpha=0.8,
    edgecolor="white",
    linewidth=1.5
)

# 3. Customise Chart
plt.title("The Sim-to-Real Gap: Overfitting in Electrical Detection", fontsize=22, fontweight='bold', pad=25, color=text_color)
plt.ylabel("Accuracy (mAP@50)", fontsize=16, color=text_color)
plt.xlabel("", fontsize=16)
plt.ylim(0, 1.1)

# Add value labels
for p in ax.patches:
    ax.annotate(f'{p.get_height():.3f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 15), 
                textcoords = 'offset points',
                fontsize=16,
                fontweight='bold',
                color=text_color)

# 4. Add "The Overfitting Mirage" Annotation
plt.annotate('The Overfitting Mirage\n(Perfect lighting, No occlusion)', 
             xy=(0, 0.974), xytext=(0.5, 1.05),
             arrowprops=dict(facecolor='white', shrink=0.05, width=1, headwidth=8),
             fontsize=12, ha='center', color=accent_color)

plt.annotate('The Reality Gap\n(Shadows, Angles, Glare)', 
             xy=(1, 0.420), xytext=(1.5, 0.6),
             arrowprops=dict(facecolor='white', shrink=0.05, width=1, headwidth=8),
             fontsize=12, ha='center', color=error_color)

# 5. Add Grid and Styling
plt.grid(axis='y', linestyle='--', alpha=0.2)
sns.despine(left=True, bottom=True)

# 6. Save Artifact
output_path = 'analysis/sim_to_real_gap.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300, transparent=False)
print(f"Visualization saved to {output_path}")

# 7. Bonus: Theoretical Accuracy Decay Plot
plt.figure(figsize=(10, 6))
epochs = np.linspace(0, 100, 100)
synth_acc = 1 - np.exp(-epochs/20)
real_acc = synth_acc * (0.4 + 0.1 * np.sin(epochs/5)) # Fluctuating/Low real accuracy

plt.plot(epochs, synth_acc, label='Synthetic Validation', color=accent_color, linewidth=3)
plt.plot(epochs, real_acc, label='Real World Generalization', color=error_color, linewidth=3, linestyle='--')
plt.fill_between(epochs, real_acc, synth_acc, color=error_color, alpha=0.1, label='Overfitting Zone')

plt.title("Evolution of Overfitting: Synthetic vs Real World", fontsize=18)
plt.xlabel("Training Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.1)
plt.savefig('analysis/overfitting_decay.png', dpi=300)
print("Decay plot saved to analysis/overfitting_decay.png")
