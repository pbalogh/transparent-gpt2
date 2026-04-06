#!/usr/bin/env python3
"""Generate figures for Darkness Visible paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ============================================================
# Figure 1: Consensus Level vs MLP Effect (ΔP)
# The paper's central finding: crossover at 3-4/7
# ============================================================
fig, ax1 = plt.subplots(figsize=(4.5, 3.2))

levels = [0, 1, 2, 3, 4, 5, 6, 7]
tokens = [210, 950, 2584, 5559, 11932, 32448, 67972, 82945]
delta_p = [0.145, 0.066, 0.041, 0.013, -0.022, -0.046, -0.058, -0.071]

colors = ['#2196F3' if d >= 0 else '#F44336' for d in delta_p]
bars = ax1.bar(levels, delta_p, color=colors, alpha=0.85, width=0.7, edgecolor='white', linewidth=0.5)

ax1.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
ax1.axvline(x=3.5, color='gray', linewidth=1.2, linestyle='--', alpha=0.7)

ax1.annotate('Crossover\n3–4/7', xy=(3.5, 0.08), fontsize=8, ha='center', 
             color='gray', fontstyle='italic')
ax1.annotate('MLP helps\n(exception path)', xy=(1, 0.12), fontsize=7.5, ha='center',
             color='#1565C0', fontstyle='italic')
ax1.annotate('MLP hurts\n(consensus path)', xy=(6, -0.055), fontsize=7.5, ha='center',
             color='#C62828', fontstyle='italic')

ax1.set_xlabel('Consensus level (of 7 neurons)')
ax1.set_ylabel('$\\Delta P$ (probability change)')
ax1.set_xticks(levels)
ax1.set_xticklabels([f'{l}/7' for l in levels])
ax1.set_ylim(-0.09, 0.18)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add token counts as secondary info
ax2 = ax1.twinx()
ax2.plot(levels, [t/1000 for t in tokens], 'o-', color='#9E9E9E', markersize=3, 
         linewidth=1, alpha=0.5)
ax2.set_ylabel('Tokens (thousands)', color='#9E9E9E', fontsize=8)
ax2.tick_params(axis='y', labelcolor='#9E9E9E', labelsize=7)
ax2.spines['top'].set_visible(False)
ax2.set_ylim(0, 100)

plt.tight_layout()
fig.savefig('fig_consensus_crossover.pdf')
fig.savefig('fig_consensus_crossover.png')
print("✓ Figure 1: Consensus crossover")
plt.close()


# ============================================================
# Figure 2: Logit Lens vs Tuned Lens Developmental Arc
# Three phases: scaffold, decision, terminal
# Both lenses show the same arc; tuned lens corrects early layers
# ============================================================
fig, ax = plt.subplots(figsize=(5.0, 3.2))

layer_labels = ['Emb', 'L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11']
x = np.arange(len(layer_labels))

# Data from tuned_vs_logit_lens.py run (204,600 tokens)
logit_top1 = [0.7, 3.0, 3.9, 3.9, 5.0, 5.9, 8.6, 10.8, 17.0, 22.7, 29.5, 34.4, 39.0]
tuned_top1 = [3.5, 9.0, 9.7, 10.7, 11.7, 12.6, 16.2, 19.0, 24.6, 29.7, 34.9, 37.3, 39.0]

# Background shading for phases
ax.axvspan(-0.5, 7.5, alpha=0.06, color='blue', label='_nolegend_')
ax.axvspan(7.5, 10.5, alpha=0.06, color='green', label='_nolegend_')
ax.axvspan(10.5, 12.5, alpha=0.06, color='red', label='_nolegend_')

# Phase boundary lines
ax.axvline(x=7.5, color='#999', linewidth=0.8, linestyle=':', alpha=0.6)
ax.axvline(x=10.5, color='#999', linewidth=0.8, linestyle=':', alpha=0.6)

# Phase labels at bottom
ax.text(3.5, 1.5, 'Scaffold', ha='center', fontsize=7, color='#666', fontstyle='italic')
ax.text(9, 1.5, 'Decision', ha='center', fontsize=7, color='#666', fontstyle='italic')
ax.text(11.5, 1.5, 'Terminal', ha='center', fontsize=7, color='#666', fontstyle='italic')

# Logit lens (dashed)
ax.plot(x, logit_top1, 'o--', color='#90CAF9', markersize=4, linewidth=1.5,
        label='Logit lens', alpha=0.8)

# Tuned lens (solid, by phase)
scaffold_color = '#1565C0'
decision_color = '#2E7D32'
terminal_color = '#C62828'

ax.plot(x[:8], tuned_top1[:8], 's-', color=scaffold_color, markersize=4.5,
        linewidth=2, label='Tuned lens — scaffold')
ax.plot(x[7:11], tuned_top1[7:11], 's-', color=decision_color, markersize=4.5,
        linewidth=2, label='Tuned lens — decision')
ax.plot(x[10:], tuned_top1[10:], 's-', color=terminal_color, markersize=4.5,
        linewidth=2, label='Tuned lens — terminal')

# Shade the gap between lenses at early layers
ax.fill_between(x[:8], logit_top1[:8], tuned_top1[:8], alpha=0.12, color='#1565C0')
ax.annotate('+6.3pp avg\ncorrection', xy=(3, 8.5), xytext=(5.5, 5),
            fontsize=6.5, ha='center', color='#1565C0', fontstyle='italic',
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=0.8, alpha=0.6))

# Convergence annotation
ax.annotate('Both 39.0%', xy=(12, 39), xytext=(12, 43),
            fontsize=6.5, ha='center', color='#555', fontstyle='italic',
            arrowprops=dict(arrowstyle='->', color='#999', lw=0.8))

ax.set_xlabel('Layer')
ax.set_ylabel('Top-1 accuracy (%)')
ax.set_xticks(x)
ax.set_xticklabels(layer_labels, fontsize=7)
ax.set_ylim(0, 48)
ax.legend(loc='upper left', fontsize=6.5, framealpha=0.95, ncol=1,
          borderpad=0.4, handlelength=1.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig('fig_logit_lens.pdf')
fig.savefig('fig_logit_lens.png')
print("✓ Figure 2: Logit lens vs tuned lens arc")
plt.close()


# ============================================================
# Figure 3: Exception Handler Tier Contribution
# Output norm vs PPL impact — the DC offset paradox
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.8))

tiers = ['Core\n(5)', 'Diff.\n(10)', 'Spec.\n(5)', 'Residual\n(~3040)']
norm_pct = [54, 23, 4, 19]  # output norm percentages
ppl_delta = [0.2, 1.3, -0.3, 2.1]  # PPL change when ablated (residual = all exception)

colors_norm = ['#1565C0', '#42A5F5', '#90CAF9', '#E0E0E0']
colors_ppl = ['#1565C0', '#42A5F5', '#90CAF9', '#E0E0E0']

# Left: Output norm contribution
bars1 = ax1.bar(range(4), norm_pct, color=colors_norm, edgecolor='white', linewidth=0.5)
ax1.set_ylabel('Exception-path output norm (%)')
ax1.set_xticks(range(4))
ax1.set_xticklabels(tiers, fontsize=7)
ax1.set_title('Magnitude', fontsize=10, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for i, v in enumerate(norm_pct):
    ax1.text(i, v + 1, f'{v}%', ha='center', fontsize=8, fontweight='bold')

# Right: PPL impact when ablated
ppl_colors = ['#1565C0', '#F44336', '#4CAF50', '#F44336']
bars2 = ax2.bar(range(4), ppl_delta, color=ppl_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_ylabel('$\\Delta$ PPL when ablated (%)')
ax2.set_xticks(range(4))
ax2.set_xticklabels(tiers, fontsize=7)
ax2.set_title('Causal impact', fontsize=10, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
for i, v in enumerate(ppl_delta):
    offset = 0.1 if v >= 0 else -0.15
    ax2.text(i, v + offset, f'{v:+.1f}%', ha='center', fontsize=8, fontweight='bold')

# Annotate the paradox
ax2.annotate('54% of norm\n→ only +0.2% PPL\n(DC offset)', 
             xy=(0, 0.2), xytext=(1.5, 1.8),
             fontsize=6.5, ha='center', fontstyle='italic', color='#555',
             arrowprops=dict(arrowstyle='->', color='#999', lw=0.8))

plt.tight_layout()
fig.savefig('fig_tier_contribution.pdf')
fig.savefig('fig_tier_contribution.png')
print("✓ Figure 3: Tier contributions")
plt.close()

print("\nAll figures generated.")
