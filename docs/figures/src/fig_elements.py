"""Render all panel elements for the two Phase-1 figures as 600-DPI PNGs.

Run after fig_data_quant.py and extract_semantic_history.py:
    uv run --no-project --with matplotlib --with numpy python docs/figures/src/fig_elements.py
Reads docs/figures/data/, writes docs/figures/elements/ (both gitignored).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRATCH = Path(__file__).resolve().parents[1] / "data"
ELEMENTS = Path(__file__).resolve().parents[1] / "elements"
ELEMENTS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.transparent": True,
    "savefig.bbox": "tight",
    "savefig.dpi": 600,
})

BLUE = "#0072B2"     # FP32
ORANGE = "#E69F00"   # INT8
GREEN = "#009E73"    # quantized / train
VERMILION = "#D55E00"  # sensitive / excluded
GREY = "#7F7F7F"

data = np.load(SCRATCH / "fig_quant_data.npz")
meta = json.loads((SCRATCH / "fig_quant_meta.json").read_text())

# ── Fig 1A: conv-group sensitivity ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(2.3, 1.9))
groups = meta["sensitivity"]
xs = np.arange(len(groups))
vals = [max(g["mse_rel_increase"] * 100, 0.05) for g in groups]
sensitive = [g["mse_rel_increase"] > 0.02 or g["agreement"] < 0.999 for g in groups]
colors = [VERMILION if s else GREEN for s in sensitive]
bars = ax.bar(xs, vals, color=colors, width=0.62)
ax.axhline(2.0, color="black", lw=0.8, ls="--")
ax.text(len(groups) - 0.42, 2.35, "exclusion\nthreshold", fontsize=7,
        ha="right", va="bottom", color="black")
for x, g, v in zip(xs, groups, vals):
    ax.text(x, v * 1.25, f"{g['agreement'] * 100:.1f}%", fontsize=7,
            ha="center", va="bottom", color="#333333")
ax.set_yscale("log")
ax.set_ylim(0.1, 400)
ax.set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
ax.set_xticks(xs, [f"{g['group']}\n(n={g['num_convs']})" for g in groups])
ax.set_xlabel("Conv group (topological order)")
ax.set_ylabel("Cursor MSE increase (%)")
fig.savefig(ELEMENTS / "q_sensitivity.png")
plt.close(fig)

# ── Fig 1B: latency ECDF ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(2.3, 1.9))
for lat, color, label in ((data["lat_fp32"], BLUE, "FP32"),
                          (data["lat_int8"], ORANGE, "INT8")):
    x = np.sort(lat)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(x, y, color=color, lw=1.4, label=label)
ax.axvline(5.0, color=GREY, lw=0.8, ls="--")
ax.axvline(10.0, color="black", lw=0.8, ls="--")
ax.text(5.0 * 0.93, 0.06, "target", fontsize=7, rotation=90,
        ha="right", va="bottom", color=GREY)
ax.text(10.0 * 0.93, 0.06, "budget", fontsize=7, rotation=90,
        ha="right", va="bottom", color="black")
ax.set_xscale("log")
ax.set_xlim(0.3, 20)
ax.set_xticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
ax.set_ylim(0, 1.02)
ax.set_xlabel("Inference latency (ms)")
ax.set_ylabel("Cumulative fraction")
# Direct line labels — a legend collides with the budget/target lines
ax.text(0.52, 0.55, "INT8", fontsize=8, color=ORANGE, ha="right")
ax.text(1.75, 0.45, "FP32", fontsize=8, color=BLUE, ha="left")
fig.savefig(ELEMENTS / "q_latency.png")
plt.close(fig)

# ── Fig 1C: cursor parity scatter ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(2.1, 1.9))
c_fp32 = data["cursor_fp32"].ravel()
c_int8 = data["cursor_int8"].ravel()
rng = np.random.default_rng(0)
idx = rng.choice(c_fp32.size, size=1500, replace=False)
lim = 0.32  # zoom to the occupied output range so deviations are visible
ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=0.8, ls="--", zorder=1)
ax.scatter(c_fp32[idx], c_int8[idx], s=3, color=ORANGE, alpha=0.35,
           linewidths=0, zorder=2, rasterized=True)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xticks([-0.3, 0, 0.3])
ax.set_yticks([-0.3, 0, 0.3])
ax.set_aspect("equal")
ax.set_xlabel("FP32 cursor output")
ax.set_ylabel("INT8 cursor output")
ax.text(0.04, 0.96,
        f"gesture agreement {meta['gesture_agreement'] * 100:.1f}%\n"
        f"cursor MSE {meta['mse_rel_increase'] * 100:+.1f}%",
        fontsize=7, ha="left", va="top", transform=ax.transAxes)
fig.savefig(ELEMENTS / "q_parity.png")
plt.close(fig)

# ── Fig 2: semantic convergence panels ───────────────────────────────────────
history = json.loads((SCRATCH / "semantic_history.json").read_text())
epochs = [h["epoch"] for h in history]


def curve_panel(fname, train_key, val_key, ylabel, extra=None, ylim=None):
    fig, ax = plt.subplots(figsize=(1.75, 1.65))
    ax.plot(epochs, [h[train_key] for h in history], color=GREEN, lw=1.3,
            label="train")
    if val_key:
        ax.plot(epochs, [h[val_key] for h in history], color=BLUE, lw=1.3,
                ls="--", label="val")
    if extra:
        extra(ax)
    ax.set_xlim(1, max(epochs))
    ax.set_xticks([1, 5, 10, 15])
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False, handlelength=1.3)
    fig.savefig(ELEMENTS / fname)
    plt.close(fig)


curve_panel("s_nll.png", "train/nll_loss", "val/nll_loss", "Gaussian NLL")
curve_panel("s_mse.png", "train/mu_mse", "val/mu_mse", "$\\mu$ MSE",
            ylim=(0, 0.055))


def chance_line(ax):
    ax.axhline(0.25, color=GREY, lw=0.8, ls=":")
    ax.text(14.6, 0.275, "chance", fontsize=7, ha="right", color=GREY)


curve_panel("s_acc.png", "train/intent_accuracy", "val/intent_accuracy",
            "Intent accuracy", extra=chance_line, ylim=(0, 1.05))

fig, ax = plt.subplots(figsize=(1.75, 1.65))
ax.plot(epochs, [h["train/intent_field_entropy"] for h in history],
        color=VERMILION, lw=1.3)
ax.set_xlim(1, max(epochs))
ax.set_xticks([1, 5, 10, 15])
ax.set_xlabel("Epoch")
ax.set_ylabel("Field entropy (nats)")
fig.savefig(ELEMENTS / "s_entropy.png")
plt.close(fig)

print("elements written:", sorted(p.name for p in ELEMENTS.glob("*.png")))
