# Update charts with the user's grid style and ensure bars are fully opaque.
import os
import matplotlib.pyplot as plt

# Global font sizes 14pt
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 14,
})

# Data
families = [r'$\pi\,0.5$', 'GR00T N1.5']
baseline = [1.0, 1.0]
quantvla = [1.22, 1.07]

# Colors
color_baseline = '#63a4f8'
color_quantvla = '#f7cd55'

# Paths
user_dir = '/home/jz97/VLM_REPO/Isaac-GR00T'
dl_dir = '/home/jz97/VLM_REPO/Isaac-GR00T'
os.makedirs(dl_dir, exist_ok=True)

def safe_save(fig, filename):
    path_dl = os.path.join(dl_dir, filename)
    fig.savefig(path_dl, dpi=500, bbox_inches='tight', facecolor='white', transparent=False)
    saved_paths = [path_dl]
    try:
        os.makedirs(user_dir, exist_ok=True)
        path_user = os.path.join(user_dir, filename)
        fig.savefig(path_user, dpi=220, bbox_inches='tight', facecolor='white', transparent=False)
        saved_paths.append(path_user)
    except Exception:
        pass
    return saved_paths

# --------- Chart 1: Grouped bars ---------
x = range(len(families))
width = 0.38

fig1 = plt.figure(figsize=(7.5, 4.6), facecolor='white')
ax1 = fig1.add_subplot(111)
ax1.bar([i - width/2 for i in x], baseline, width=width, label='Baseline', color=color_baseline, alpha=1.0)
ax1.bar([i + width/2 for i in x], quantvla, width=width, label='QuantVLA', color=color_quantvla, alpha=1.0)

ax1.set_ylabel('Inference speedup')
ax1.set_xticks(list(x))
ax1.set_xticklabels(families)
ax1.set_ylim(0.9, 1.3)
ax1.grid(True, axis="y", alpha=0.18, linestyle="--")
ax1.legend(framealpha=1.0, facecolor='white', edgecolor='black')

for i, v in enumerate(baseline):
    ax1.text(i - width/2, v + 0.01, f'{v:.2f}x', ha='center', va='bottom')
for i, v in enumerate(quantvla):
    ax1.text(i + width/2, v + 0.01, f'{v:.2f}x', ha='center', va='bottom')

fig1.tight_layout()
paths_grouped = safe_save(fig1, 'quantvla_speedup_grouped_v4.png')
plt.show()

# --------- Chart 2: All variants ---------
labels = [r'$\pi\,0.5$ Baseline', r'$\pi\,0.5$ + QuantVLA (W4A8)', 'GR00T N1.5 Baseline', 'GR00T N1.5 + QuantVLA (W4A8)']
values = [1.0, 1.22, 1.0, 1.07]
colors = [color_baseline, color_quantvla, color_baseline, color_quantvla]

fig2 = plt.figure(figsize=(9, 4.6), facecolor='white')
ax2 = fig2.add_subplot(111)
bars = ax2.bar(range(len(labels)), values, color=colors, alpha=1.0)

ax2.set_ylabel('Inference speedup (×)')
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels, rotation=12, ha='right')
ax2.set_ylim(0.9, 1.3)
ax2.grid(True, axis="y", alpha=0.18, linestyle="--")
ax2.legend(handles=[bars[0], bars[1]], labels=['Baseline', 'QuantVLA'], framealpha=1.0, facecolor='white', edgecolor='black')

for i, v in enumerate(values):
    ax2.text(i, v + 0.01, f'{v:.2f}x', ha='center', va='bottom')

fig2.tight_layout()
paths_all = safe_save(fig2, 'quantvla_speedup_all_v4.png')
plt.show()

paths_grouped, paths_all
