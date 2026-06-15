import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.metrics import confusion_matrix
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_PATH  = Path(__file__).parent / "C:/Uni/Project/BlameBERT/data/training_data/validation_set/predicted_val_data.jsonl"
#OUTPUT_PDF = Path(__file__).parent / "confusion_matrices_by_party.pdf"
OUTPUT_PNG = Path(__file__).parent / "confusion_matrices_by_party.png"

MIN_N = 10     # minimum sentences per party to include
VMAX = 0.75   # colormap upper bound (proportion of total)
NCOLS = 2      # parties per row
COL_WIDTH = 3.5    # inches per column
ROW_HEIGHT = 3.2    # inches per row

# ── Load data ─────────────────────────────────────────────────────────────────

if not Path(DATA_PATH).exists():
    raise FileNotFoundError(
        f"Data file not found: {DATA_PATH}\n"
        "Update DATA_PATH at the top of this script."
    )

with open(DATA_PATH) as f:
    df = pd.DataFrame([json.loads(line) for line in f])

print(f"Loaded {len(df)} rows")

# ── Select parties ────────────────────────────────────────────────────────────

parties = df["party"].value_counts()
parties = parties[parties >= MIN_N].index.tolist()
print(f"Plotting {len(parties)} parties: {parties}")

# ── Colormap ──────────────────────────────────────────────────────────────────
# Change the two hex values to adjust the color ramp.
# Current: white -> blue. Try e.g. ['#fff5eb', '#d94801'] for orange.

cmap = LinearSegmentedColormap.from_list("blame", ["#f7fbff", "#2171b5"])
norm = Normalize(vmin=0, vmax=VMAX)

# ── Layout ────────────────────────────────────────────────────────────────────

nrows = int(np.ceil((len(parties) + 1) / NCOLS))  # +1 reserves a cell for summary
fig = plt.figure(figsize=(NCOLS * COL_WIDTH, nrows * ROW_HEIGHT))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(nrows, NCOLS, figure=fig, hspace=0.3, wspace=0.01)

axes = []
for i in range(len(parties)):
    row, col = divmod(i, NCOLS)
    axes.append(fig.add_subplot(gs[row, col]))

summary_row, summary_col = divmod(len(parties), NCOLS)
ax_summary = fig.add_subplot(gs[summary_row, summary_col])
ax_summary.axis("off")

# ── Draw each party panel using patches (backend-agnostic) ───────────────────

for ax, party in zip(axes, parties):
    sub = df[df["party"] == party]
    n   = len(sub)

    cm      = confusion_matrix(sub["label"], sub["prediction"], labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum()

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    for r in range(2):
        for c in range(2):
            pct   = cm_norm[r, c]
            count = cm[r, c]
            color = cmap(norm(pct))
            rect  = mpatches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                linewidth=0.5, edgecolor="#cccccc", facecolor=color,
            )
            ax.add_patch(rect)
            text_color = "white" if pct > 0.38 else "#1a1a1a"
            ax.text(
                c, r,
                f"{count}\n({pct * 100:.0f}%)",
                ha="center", va="center",
                fontsize=10, color=text_color,
                linespacing=1.4,
            )

    err = (cm[0, 1] + cm[1, 0]) / n
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: 0", "Pred: 1"], fontsize=10)
    ax.set_yticklabels(["True: 0", "True: 1"], fontsize=10, rotation=90, va="center")
    ax.tick_params(length=0)
    ax.set_title(f"{party}  (n={n})", fontsize=9.5, fontweight="bold")#, pad=6)
    ax.set_xlabel(f"error rate: {err * 100:.0f}%", fontsize=7.5,
                  color="#555555", labelpad=4)

# ── Shared colorbar ───────────────────────────────────────────────────────────

cbar_ax = fig.add_axes([0.93, 0.35, 0.02, 0.30])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
#cbar.set_label("Proportion of total", fontsize=8, labelpad=8)
cbar.ax.tick_params(labelsize=7.5)

# ── Title & save ──────────────────────────────────────────────────────────────
plt.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=300, facecolor="white")
print(f"Saved {OUTPUT_PNG}")