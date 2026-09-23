"""
Regenerates the headline divergence figure (Telco vs Adult) at high resolution.
Reads directly from comparisons_long.csv so the figure always matches the data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("results/comparisons_long.csv")

metric = "spearman"  # the metric shown on the slide
lambdas = [0.0, 0.5, 1.0, 1.5, 2.0]


def curve(dataset):
    sub = df[(df.dataset == dataset) & (df.metric == metric)]
    return [sub[sub["lambda"] == l]["cliffs_delta"].mean() for l in lambdas]


telco = curve("telco")
adult = curve("adult")

# ---- figure ----
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)  # high-res

TELCO_C = "#E76F51"  # coral
ADULT_C = "#2A6F97"  # blue

ax.axhline(0, color="#999999", linewidth=1, zorder=1)  # zero line
ax.plot(
    lambdas,
    telco,
    "-o",
    color=TELCO_C,
    linewidth=2.5,
    markersize=7,
    label="Telco (inverts)",
    zorder=3,
)
ax.plot(
    lambdas,
    adult,
    "-o",
    color=ADULT_C,
    linewidth=2.5,
    markersize=7,
    label="Adult (positive, no inversion)",
    zorder=3,
)

#  Shade Adult's ESS-limited region (lambda >= 1.5)
ax.axvspan(1.25, 2.05, color="#cccccc", alpha=0.15, zorder=0)
ax.text(1.65, -0.27, "Adult ESS-limited", fontsize=8, color="#777777", ha="center")

ax.set_xlabel("Shift intensity  λ", fontsize=12)
ax.set_ylabel("Cliff's δ   (+ = confident more stable)", fontsize=11)
ax.set_ylim(-0.3, 0.3)
ax.set_xticks(lambdas)
ax.set_yticks(np.arange(-0.3, 0.31, 0.1))
ax.grid(axis="y", color="#E4E8F0", linewidth=1)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=10, loc="upper right")

plt.tight_layout()
plt.savefig("figures/divergence.png", dpi=300, bbox_inches="tight")
plt.savefig("figures/divergence.pdf", bbox_inches="tight")  # vector, for the report
print("saved figures/divergence.png and figures/divergence.pdf")
