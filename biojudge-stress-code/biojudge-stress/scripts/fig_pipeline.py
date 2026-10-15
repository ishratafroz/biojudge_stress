"""Draw Fig. 1 (pipeline diagram) as a PNG for the Word version of the paper."""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BOXES = [
    ("1. Items", "PubMedQA (literature)\nMedQA (clinical)\nMedMCQA (pharmacology)\n+ answer key"),
    ("2. Solo answers", "4 families answer\neach item alone\n-> hard-negative\ndistractor"),
    ("3. Author agents", "each family writes\nR+ (keyed answer)\nR- (distractor)\nvalidated, 80-150 words"),
    ("4. Perturbation", "padding (utility agent)\n6 content-free edits\nnormalizer agent\n(correction)"),
    ("5. Verifier agents", "pairwise: 4 judges\nx order x protocol\nbest-of-4 selector\n(base / debias)"),
    ("6. Aggregation", "single, swap,\njury, recusal jury,\nfull pipeline"),
    ("7. Score vs. key", "Acc, kappa, Flip, PPI,\nVB, SPB, IC, ASR,\ncapture"),
]
FILL = ["#dbe8f5", "#dbe8f5", "#dbe8f5", "#f8e3cf", "#e1f0dc", "#e1f0dc", "#ece2f3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figs/fig_pipeline.png")
    a = ap.parse_args()
    n = len(BOXES)
    w, h, gap = 1.8, 1.2, 0.3
    fig, ax = plt.subplots(figsize=(n * (w + gap) * 0.95, 1.75))
    for i, ((title, body), col) in enumerate(zip(BOXES, FILL)):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=col, ec="#444444", lw=0.8))
        ax.text(x + w / 2, h - 0.16, title, ha="center", va="center", fontsize=8.2, weight="bold")
        ax.text(x + w / 2, h / 2 - 0.1, body, ha="center", va="center", fontsize=6.6, linespacing=1.25)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.03, h / 2), (x + w + gap - 0.03, h / 2),
                                         arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#333333"))
    ax.set_xlim(-0.05, n * (w + gap) - gap + 0.05)
    ax.set_ylim(-0.05, h + 0.05)
    ax.axis("off")
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
