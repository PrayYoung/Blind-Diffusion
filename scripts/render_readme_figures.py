"""Render README SVG/PNG figures from curated public counts (requires matplotlib)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = '#182438', '#61718a', '#e4eaf2'
PURPLE, TEAL, ORANGE, GRAY = '#7245c7', '#008c95', '#df8937', '#7a8799'


def save(fig, destination, name):
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ('svg', 'png'):
        fig.savefig(destination / f'{name}.{extension}', dpi=200, facecolor='white')
    plt.close(fig)


def blackout(destination):
    data = json.loads((ROOT / 'docs/results/matched_summary.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    fig.subplots_adjust(left=.065, right=.978, top=.68, bottom=.20, wspace=.14)
    fig.text(.065, .94, 'When observations stop, belief keeps control moving',
             fontsize=20, weight='bold', color=INK)
    fig.text(.065, .885, 'Matched starts · one training seed · permanent blackout after five visible observations at D=20',
             fontsize=10.2, color=MUTED)
    style = {
        'cfm4': (PURPLE, 'o', '-', 2.8, 'CFM-4 + prior · ours'),
        'cfm2': ('#b49adb', 's', '--', 1.5, 'CFM-2 + prior · ours (Lift)'),
        'ddim10': (TEAL, 'D', '-', 2.3, 'DDIM-10 + prior · ours'),
        'open_loop': (ORANGE, '^', '--', 1.9, 'Zero-input DDIM chunks · ablation'),
        'bc': (GRAY, 'v', ':', 1.9, 'Reactive BC · baseline'),
        'mpc': ('#b3beca', 'x', '-.', 1.4, 'Latent MPC · baseline (Lift)'),
    }
    for ax, task in zip(axes, ('lift', 'can')):
        rows = data['lift']['results'] if task == 'lift' else data['can']['results_by_horizon']['300']
        ax.set_title('Lift  /  50 seeds · 500 steps' if task == 'lift' else 'Can  /  30 seeds · 300 steps',
                     loc='left', fontsize=13, weight='bold', color=INK, pad=14)
        ax.axvspan(94, 105, color='#f3effa', zorder=0)
        for method in reversed(list(style)):
            if method not in rows:
                continue
            color, marker, line, width, _ = style[method]
            values = [100 * rows[method][str(d)]['rate'] for d in (0, 5, 10, 15, 20)]
            ax.plot((0, 25, 50, 75, 100), values, color=color, marker=marker,
                    linestyle=line, linewidth=width, markersize=6,
                    markerfacecolor='white' if method in ('ddim10', 'cfm2', 'open_loop', 'bc') else color,
                    markeredgewidth=1.5, clip_on=False, zorder=5 if method == 'cfm4' else 3)
        ax.set(xlim=(-3, 104), ylim=(-4, 106), xticks=(0, 25, 50, 75, 100), yticks=(0, 25, 50, 75, 100))
        ax.set_xticklabels(['0\nClean', '25', '50', '75', '100\nPermanent'])
        ax.set_xlabel('Scheduled observation blackout (%)', labelpad=8, fontsize=10, color=MUTED)
        ax.grid(axis='y', color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=7)
        for edge in ax.spines.values():
            edge.set_visible(False)
    axes[0].set_ylabel('Task success (%)', color=MUTED, fontsize=10, labelpad=9)
    handles = [Line2D([], [], color=c, marker=m, linestyle=l, linewidth=w, markersize=5, label=label)
               for c, m, l, w, label in style.values()]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.055, .85), ncol=3,
               frameon=False, fontsize=9, labelcolor=INK, columnspacing=2.2, handlelength=2.7)
    fig.text(.065, .045, 'Points are observed rates; lines connect measured conditions. Wilson 95% intervals and all counts are linked below.',
             color=MUTED, fontsize=9)
    save(fig, destination, 'blackout_comparison')


def architecture(destination):
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 100), ylim=(0, 54))
    ax.axis('off')

    def text(x, y, label, size=10, color=INK, weight='normal', ha='left'):
        ax.text(x, y, label, fontsize=size, color=color, weight=weight, ha=ha,
                va='center', linespacing=1.5)

    def box(x, y, w, h, face, edge='#dce3ed', radius=.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                                   facecolor=face, edgecolor=edge, linewidth=1.1, zorder=1))

    def arrow(start, end, color=MUTED, rad=0, style='-', zorder=2):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=13,
                                    linewidth=1.4, color=color, linestyle=style,
                                    connectionstyle=f'arc3,rad={rad}', zorder=zorder))

    text(3, 50.5, 'Blind-Diffusion  /  one causal control tick', 21, weight='bold')
    text(3, 46.8, 'Observe when available. Propagate belief when blind. Generate a plan. Execute one action.', 11, MUTED)
    box(3, 10, 44, 32, '#f8f5fd', '#ded3f0')
    box(50, 10, 26, 32, '#f0f6fd', '#d1e0f3')
    box(79, 10, 18, 32, '#f0f9f6', '#cfe5dc')
    text(6, 39, '01   CAUSAL MEMORY', 11, PURPLE, 'bold')
    text(53, 39, '02   FLOW POLICY', 11, '#2563aa', 'bold')
    text(82, 39, '03   EXECUTION', 11, '#21775a', 'bold')

    box(7, 30, 36, 6.3, 'white', '#d5c7ec')
    text(25, 34.3, 'Shared recurrent transition', 11, weight='bold', ha='center')
    text(25, 31.9, r'$h_t = \mathrm{GRU}(h_{t-1}, [z_{t-1}, a_{t-1}])$', 11, ha='center')
    box(6, 20, 18, 6.2, 'white', '#d5c7ec')
    box(26, 20, 18, 6.2, '#eae0f8', '#bba0df')
    arrow((17, 30), (15, 26.3), PURPLE)
    arrow((33, 30), (35, 26.3), PURPLE)
    text(8, 28.3, 'VISIBLE', 8.5, MUTED, 'bold')
    text(36, 28.3, 'BLACKOUT', 8.5, PURPLE, 'bold')
    text(15, 24.2, 'Posterior mean', 10.5, weight='bold', ha='center')
    text(15, 21.9, 'Uses normalized, encoded oₜ', 8.2, MUTED, ha='center')
    text(35, 24.2, 'Prior mean', 10.5, PURPLE, 'bold', ha='center')
    text(35, 21.9, 'No observation encoded', 8.5, PURPLE, ha='center')
    box(14, 12, 22, 5.3, 'white', '#bba0df')
    text(25, 15.5, 'Pre-action belief', 10.2, weight='bold', ha='center')
    text(25, 13.4, r'$b_t = [h_t, z_t]$', 11, ha='center')
    arrow((15, 20), (21, 17.3), PURPLE)
    arrow((35, 20), (29, 17.3), PURPLE)

    box(53, 30, 20, 6.3, 'white', '#c1d6ef')
    text(63, 34.2, 'Fresh Gaussian action noise', 9.5, weight='bold', ha='center')
    text(63, 31.9, '8 actions × 7 native coordinates', 8.5, MUTED, ha='center')
    box(53, 20, 20, 7, '#dceafb', '#a9c7e9')
    text(63, 24.9, 'Conditional flow field', 11, '#2563aa', 'bold', ha='center')
    text(63, 22.4, 'FiLM U-Net · 4 Euler evaluations', 8.9, '#2563aa', ha='center')
    arrow((63, 30), (63, 27), '#2563aa')
    # Belief conditioning enters the flow field, not the initial Gaussian noise.
    arrow((36, 14.7), (53, 23), PURPLE, rad=.12)
    text(46, 19, 'condition', 8.4, PURPLE, ha='center')
    box(53, 12, 20, 5.3, 'white', '#c1d6ef')
    text(63, 15.5, 'Completed 8-action plan', 9.8, weight='bold', ha='center')
    text(63, 13.4, 'Clamp endpoint to [−1, 1]', 8.8, MUTED, ha='center')
    arrow((63, 20), (63, 17.3), '#2563aa')

    box(81, 25, 14, 10, 'white', '#b7d8c9')
    text(88, 32.5, 'Execute first action', 10.3, '#21775a', 'bold', ha='center')
    text(88, 29.7, '7-D robot command', 9, MUTED, ha='center')
    text(88, 27.3, 'Discard the remaining 7', 8.5, MUTED, ha='center')
    arrow((73, 14.6), (81, 28), '#2563aa', rad=.12)
    box(81, 12, 14, 7, 'white', '#b7d8c9')
    text(88, 16.9, 'Environment step', 10.2, weight='bold', ha='center')
    text(88, 14.2, 'Lift / Can · MuJoCo', 8.8, MUTED, ha='center')
    arrow((88, 25), (88, 19), '#21775a')

    # Feedback carries the selected belief and actual executed action.
    ax.plot([88, 88, 5, 5], [12, 6.6, 6.6, 33.2], color=MUTED, linewidth=1.3, linestyle=(0, (4, 3)), zorder=0)
    arrow((5, 33.2), (7, 33.2), MUTED, style='--')
    text(49, 6.6, '  NEXT TICK: carry (hₜ, zₜ) and the executed action aₜ; gate the next observation  ',
         9.3, MUTED, ha='center')
    # An opaque label prevents the feedback rule from crossing its text.
    ax.text(49, 6.6, '  NEXT TICK: carry (hₜ, zₜ) and the executed action aₜ; gate the next observation  ',
            fontsize=9.3, color=MUTED, ha='center', va='center', backgroundcolor='white')
    text(3, 2.4, 'Inference uses latent means. This diagram covers CFM-4 receding-horizon control; DDIM, zero-input chunks and MPC are separate comparison variants.',
         8.8, MUTED)
    save(fig, destination, 'system_architecture')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'assets/figures')
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'axes.unicode_minus': False})
    blackout(args.output)
    architecture(args.output)


if __name__ == '__main__':
    main()
