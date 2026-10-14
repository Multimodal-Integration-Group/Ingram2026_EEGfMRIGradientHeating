# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "pandas",
# ]
# ///

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).resolve().parents[1]

plt.rcParams["font.family"] = "Arial"

df = pd.read_csv(
    root / "data" / "7T_temperature_recordings" / "7T_heating_slew_rates.csv"
)

df["time_s"] = df["time"] * 60

slew_cols = [c for c in df.columns if c not in ("time", "time_s")]

fig, axes = plt.subplots(
    1, len(slew_cols), figsize=(3 * len(slew_cols), 5), sharey=True
)

color = "#7570b3"

for i, (ax, col) in enumerate(zip(axes, slew_cols)):
    ax.plot(
        df["time_s"],
        df[col],
        color=color,
        linestyle="-",
        linewidth=2,
    )

    ax.set_xlim(left=0)
    ax.set_ylim(-0.5, 2.5)

    ax.set_title(col, fontsize=18)
    ax.set_xlabel("Time (s)", fontsize=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.5)
    ax.spines["bottom"].set_linewidth(2.5)
    ax.tick_params(labelsize=12, width=2.5)

    ax.text(
        -0.15,
        1.05,
        chr(65 + i) + ")",
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
    )

axes[0].set_ylabel("ΔTemperature (°C)", fontsize=16)

plt.tight_layout()
plt.savefig(root / "figures" / "figure_03.svg", dpi=300, bbox_inches="tight")
plt.savefig(root / "figures" / "figure_03.png", dpi=300, bbox_inches="tight")
