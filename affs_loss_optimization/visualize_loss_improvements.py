"""
Generate a visual comparison diagram of old vs new loss function
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_loss_comparison_diagram():
    fig = plt.figure(figsize=(16, 12))

    # Create a grid layout
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

    # =========================================================================
    # Panel 1: Focal Loss vs BCE
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    p_t = np.linspace(0.01, 0.99, 100)
    bce = -np.log(p_t)
    focal_gamma2 = -((1 - p_t) ** 2) * np.log(p_t)
    focal_gamma5 = -((1 - p_t) ** 5) * np.log(p_t)

    ax1.plot(p_t, bce, 'b-', linewidth=2, label='BCE (old)')
    ax1.plot(p_t, focal_gamma2, 'r-', linewidth=2, label='Focal γ=2 (new)')
    ax1.plot(p_t, focal_gamma5, 'r--', linewidth=2, label='Focal γ=5 (aggressive)')
    ax1.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.fill_between([0, 0.5], 0, ax1.get_ylim()[1], alpha=0.1, color='red', label='Hard examples')
    ax1.fill_between([0.5, 1], 0, ax1.get_ylim()[1], alpha=0.1, color='green', label='Easy examples')

    ax1.set_xlabel('Predicted Probability (p)', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('1. Focal Loss: Focus on Hard Examples', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 5)

    # Add annotation
    ax1.annotate(
        'Downweights\neasy examples',
        xy=(0.8, 0.5),
        xytext=(0.65, 2),
        arrowprops=dict(arrowstyle='->', color='green', lw=2),
        fontsize=10,
        color='green',
        fontweight='bold',
    )
    ax1.annotate(
        'Upweights\nhard examples',
        xy=(0.2, 3),
        xytext=(0.35, 4),
        arrowprops=dict(arrowstyle='->', color='red', lw=2),
        fontsize=10,
        color='red',
        fontweight='bold',
    )

    # =========================================================================
    # Panel 2: Class Imbalance
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    categories = ['Background\n(90%)', 'Foreground\n(10%)']
    bce_contribution = [0.7, 0.3]
    focal_contribution = [0.3, 0.7]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax2.bar(x - width / 2, bce_contribution, width, label='BCE (old)', color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x + width / 2, focal_contribution, width, label='Focal (new)', color='coral', alpha=0.8)

    ax2.set_ylabel('Contribution to Gradient', fontsize=11)
    ax2.set_title('2. Handling Class Imbalance', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    # =========================================================================
    # Panel 3: Boundary Emphasis
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])

    # Create a simple segmentation mask
    size = 50
    mask = np.zeros((size, size))
    mask[15:35, 15:35] = 1

    # Create boundary
    from scipy.ndimage import sobel

    boundary = np.abs(sobel(mask, axis=0)) + np.abs(sobel(mask, axis=1)) > 0

    # Create weight map
    weight_map = np.ones_like(mask)
    weight_map[boundary] = 2.0

    im = ax3.imshow(weight_map, cmap='YlOrRd', vmin=1, vmax=2)
    ax3.contour(mask, colors='blue', linewidths=2)
    plt.colorbar(im, ax=ax3, label='Loss Weight', fraction=0.046)
    ax3.set_title('3. Boundary Emphasis (2× weight)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Interior: 1×     Boundary: 2×', fontsize=10)
    ax3.set_xticks([])
    ax3.set_yticks([])

    # =========================================================================
    # Panel 4: LSDS Component Weighting
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])

    components = ['Offset\n(3 ch)', 'Variance\n(3 ch)', 'Pearson\n(3 ch)', 'Mass\n(1 ch)']
    old_weights = [1.0, 1.0, 1.0, 1.0]
    new_weights = [2.0, 1.5, 1.0, 1.0]

    x = np.arange(len(components))
    width = 0.35

    bars1 = ax4.bar(x - width / 2, old_weights, width, label='Old (equal)', color='steelblue', alpha=0.8)
    bars2 = ax4.bar(x + width / 2, new_weights, width, label='New (weighted)', color='coral', alpha=0.8)

    ax4.set_ylabel('Loss Weight', fontsize=11)
    ax4.set_title('4. LSDS Component Weighting', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(components, fontsize=9)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 2.5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.05,
                f'{height:.1f}×',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
            )

    # =========================================================================
    # Panel 5: Affinity Channel Weighting
    # =========================================================================
    ax5 = fig.add_subplot(gs[2, :])

    channels = [
        'Direct\n[-1,0,0]',
        'Direct\n[0,-1,0]',
        'Direct\n[0,0,-1]',
        'Long\n[-3,0,0]',
        'Long\n[0,-3,0]',
        'Long\n[0,0,-3]',
        'Diag\n[-1,-1,0]',
        'Diag\n[-1,0,-1]',
        'Diag\n[0,-1,-1]',
    ]
    old_weights = [1.0] * 9
    new_weights = [1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 0.85, 0.85, 0.85]

    colors = ['#2E86AB'] * 3 + ['#A23B72'] * 3 + ['#F18F01'] * 3

    x = np.arange(len(channels))
    width = 0.35

    bars1 = ax5.bar(
        x - width / 2, old_weights, width, label='Old (equal)', color='lightgray', alpha=0.6, edgecolor='black'
    )
    bars2 = ax5.bar(
        x + width / 2, new_weights, width, label='New (weighted)', color=colors, alpha=0.8, edgecolor='black'
    )

    ax5.set_ylabel('Loss Weight', fontsize=11)
    ax5.set_title('5. Affinity Channel Weighting', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(channels, fontsize=8, rotation=0)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 1.3)

    # Add group labels
    ax5.text(1, 1.15, 'Direct Neighbors\n(Most Reliable)', ha='center', fontsize=9, fontweight='bold', color='#2E86AB')
    ax5.text(4, 1.15, 'Long-Range\n(Noisy but Needed)', ha='center', fontsize=9, fontweight='bold', color='#A23B72')
    ax5.text(7, 1.15, 'Diagonal\n(Moderate)', ha='center', fontsize=9, fontweight='bold', color='#F18F01')

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=8,
            fontweight='bold',
        )

    # =========================================================================
    # Panel 6: Expected Improvements
    # =========================================================================
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')

    improvements = [
        ('False Positives', '5-15%', 'Focal loss penalizes confident wrong predictions'),
        ('Holes in Objects', '10-20%', 'Offset LSDS channels get 2× gradient'),
        ('Over-Merging', '20-30%', 'Boundary emphasis at contact points'),
        ('Over-Splitting', '5-10%', 'Appropriate long-range affinity weighting'),
    ]

    y_start = 0.85
    for i, (problem, improvement, reason) in enumerate(improvements):
        y = y_start - i * 0.22

        # Problem box
        box1 = FancyBboxPatch(
            (0.02, y - 0.08), 0.18, 0.15, boxstyle="round,pad=0.01", edgecolor='black', facecolor='#FFE5E5', linewidth=2
        )
        ax6.add_patch(box1)
        ax6.text(0.11, y, problem, ha='center', va='center', fontsize=11, fontweight='bold')

        # Arrow
        arrow = FancyArrowPatch((0.22, y), (0.30, y), arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
        ax6.add_patch(arrow)

        # Improvement box
        box2 = FancyBboxPatch(
            (0.32, y - 0.08), 0.12, 0.15, boxstyle="round,pad=0.01", edgecolor='green', facecolor='#E5F5E5', linewidth=2
        )
        ax6.add_patch(box2)
        ax6.text(0.38, y, f'↓ {improvement}', ha='center', va='center', fontsize=12, fontweight='bold', color='green')

        # Reason
        ax6.text(0.47, y, reason, ha='left', va='center', fontsize=10, style='italic')

    ax6.text(0.5, 0.98, 'Expected Improvements', ha='center', va='top', fontsize=14, fontweight='bold')

    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)

    # =========================================================================
    # Main title
    # =========================================================================
    fig.suptitle('Loss Function Optimization for Mitochondria Segmentation', fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig('loss_optimization_visual.png', dpi=200, bbox_inches='tight', facecolor='white')
    print("✓ Visualization saved to: loss_optimization_visual.png")

    return fig


if __name__ == "__main__":
    create_loss_comparison_diagram()
    print("\nVisualization created successfully!")
    print("This diagram shows:")
    print("  1. How focal loss focuses on hard examples")
    print("  2. How it handles class imbalance")
    print("  3. Boundary emphasis visualization")
    print("  4. LSDS component weighting")
    print("  5. Affinity channel weighting")
    print("  6. Expected improvements in segmentation")
