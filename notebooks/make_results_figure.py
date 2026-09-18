"""
Draws results/progress.png: best value found so far, per round, one panel per function.

Small multiples rather than one chart, because the eight functions span very
different scales (Function 5 reaches ~7000, Function 1 started near 1e-16) and
putting them on shared axes would hide everything except Function 5.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, LINE = "#1a1a1a", "#6b6b6b", "#2b5c8a"

fig, axes = plt.subplots(2, 4, figsize=(13, 6))
for i, ax in enumerate(axes.flat, start=1):
    y = np.ravel(np.load(f"data/function_{i}/outputs.npy"))
    start = len(y) - 7                      # 7 rounds submitted so far
    running = np.maximum.accumulate(y)      # best found up to each reading
    rounds = np.arange(0, len(y) - start + 1)
    series = np.concatenate([[running[start - 1]], running[start:]])

    ax.plot(rounds, series, color=LINE, linewidth=2,
            marker="o", markersize=5, markerfacecolor=LINE, markeredgecolor="white")
    ax.set_title(f"Function {i}", fontsize=11, color=INK, pad=8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d0d0d0")
    # Function 1 spans many orders of magnitude, so it needs a log axis to show anything
    if i == 1:
        ax.set_yscale("log")
        ax.set_title("Function 1  (log scale)", fontsize=11, color=INK, pad=8)
    ax.set_xlabel("round", fontsize=9, color=MUTED)

fig.suptitle("Best value found so far, by round", fontsize=13, color=INK, y=0.98)
fig.text(0.5, 0.005, "round 0 is the starting data before any query was submitted",
         ha="center", fontsize=9, color=MUTED)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("results/progress.png", dpi=150, facecolor="white")
print("wrote results/progress.png")
