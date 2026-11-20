"""Render the public Lift figure from the curated, non-sensitive summary."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=root / "docs/results/lift_flow_matching.json")
    parser.add_argument("--output", type=Path, default=root / "assets/lift_results")
    args = parser.parse_args()
    evaluation = json.loads(args.data.read_text())["evaluation"]
    n = evaluation["episodes_per_condition"]
    colors = {"clean": "#3665DB", "blind": "#008874"}
    ink, muted, background = "#14243B", "#526178", "#F4F7FC"
    for mode in colors:
        assert sum(evaluation[mode]["success_by_seed"]) == evaluation[mode]["successes"]
        assert len(evaluation[mode]["success_by_seed"]) == n
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "blind-diffusion-lift-v2",
                         "axes.unicode_minus": False})
    fig = plt.figure(figsize=(12, 6), facecolor=background)
    fig.text(.055, .925, "Lift control after observations stop", fontsize=25, weight="bold", color=ink)
    fig.text(.055, .872, "OT-path CFM  /  4 Euler steps  /  20 paired evaluation seeds", fontsize=12, color=muted)
    ax = fig.add_axes([.08, .23, .35, .55], facecolor=background)
    ax.set_ylim(0, 112)
    ax.set_xlim(-.6, 1.6)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="both", length=0, labelcolor=muted, pad=8)
    ax.grid(axis="y", color="#DFE6F1", linewidth=.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, mode in enumerate(colors):
        count = evaluation[mode]["successes"]
        rate = 100 * count / n
        ax.bar(i, rate, width=.55, color=colors[mode], zorder=2)
        ax.text(i, rate + 3, f"{rate:.0f}%", ha="center", fontsize=23, weight="bold", color=colors[mode])
        ax.text(i, rate - 11, f"{count}/{n}", ha="center", fontsize=13, color="white", weight="bold")
    ax.set_xticks([0, 1], ["Clean", "Permanently blind"])
    ax.set_ylabel("Task success", color=muted, labelpad=10)
    tile = fig.add_axes([.51, .49, .44, .24])
    tile.set_xlim(-2.8, n)
    tile.set_ylim(-1.0, 2.3)
    tile.axis("off")
    tile.text(-2.8, 2.05, "Same starts. Every outcome shown.", fontsize=13, weight="bold", color=ink)
    for row, mode in enumerate(("clean", "blind")):
        y = 1.1 - row
        tile.text(-.6, y + .25, mode.capitalize(), ha="right", va="center", color=muted, fontsize=10)
        for i, success in enumerate(evaluation[mode]["success_by_seed"]):
            tile.add_patch(Rectangle((i + .08, y), .73, .52,
                                    facecolor=colors[mode] if success else "#FBE7E3", edgecolor="none"))
            if not success:
                tile.text(i + .445, y + .27, "×", ha="center", va="center", color="#B84235", fontsize=12)
    for i in evaluation["seeds"]:
        tile.text(i + .445, -.28, str(i), ha="center", fontsize=7.5, color=muted)
    tile.text(9.5, -.75, "Evaluation seed", ha="center", fontsize=9, color=muted)
    protocol = fig.add_axes([.51, .22, .44, .21])
    protocol.axis("off")
    protocol.text(0, .95, "The blind condition", fontsize=13, weight="bold", color=ink)
    for x, width, color, title, subtitle in (
        (0, .29, "#E4ECFC", "5 visible", "steps 0–4"),
        (.36, .64, "#DDF1EB", "No new observations", "steps 5–499"),
    ):
        protocol.add_patch(FancyBboxPatch((x, .18), width, .52, boxstyle="round,pad=0.012,rounding_size=.03",
                                        facecolor=color, edgecolor="none"))
        protocol.text(x + width/2, .5, title, ha="center", color=ink, weight="bold", fontsize=11)
        protocol.text(x + width/2, .29, subtitle, ha="center", color=muted, fontsize=9)
    protocol.annotate("", xy=(.35, .44), xytext=(.30, .44), arrowprops={"arrowstyle": "->", "color": muted})
    protocol.text(0, -.08, "Remembered belief + executed actions → next belief", fontsize=10, color=muted)
    fig.text(.055, .075, "RoboMimic Lift · one training seed · task predicate reached within 500 steps", fontsize=10, color=muted)
    fig.text(.055, .038, "Paired initial observations verified exactly. Success does not imply a sustained hold.", fontsize=9, color=muted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        output_path = args.output.with_suffix(f".{suffix}")
        fig.savefig(output_path, dpi=180, facecolor=background,
                    metadata={"Date": None} if suffix == "svg" else None)
        if suffix == "svg":
            output_path.write_text("\n".join(line.rstrip() for line in output_path.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
