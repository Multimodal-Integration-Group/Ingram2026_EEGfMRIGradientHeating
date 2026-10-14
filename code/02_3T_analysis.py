# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "arviz",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pathlib",
#     "scipy",
# ]
# ///
import os
import re
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

root = Path(__file__).resolve().parents[1]
temperature_data = Path(root / "data" / "3T_temperature_recordings")
plt.rcParams["font.family"] = "Arial"

probe_labels = {
    "Probe_1": "FT10",
    "Probe_2": "FT7",
    "Probe_3": "Fp1",
    "Probe_4": "ECG",
}

### Analysis Functions ###


def parse_temperature_log(file_path):
    records = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            temps = [float(v) for v in re.findall(r"(\d+\.\d+)\s+C", line)]
            match = re.search(r"(\d{2}:\d{2}:\d{2}) (\d{2}/\d{2}/\d{4})", line)
            timestamp = f"{match[2]} {match[1]}" if match else None
            records.append(temps + [timestamp])
    df = pd.DataFrame(
        records, columns=["Probe_1", "Probe_2", "Probe_3", "Probe_4", "Timestamp"]
    )
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    df["Time_s"] = df.index.astype(float)
    return df


def baseline_correct_temperature(
    df, epi_start, epi_end, baseline_points=5, probes=None
):
    if probes is None:
        probes = ["Probe_1", "Probe_2", "Probe_3", "Probe_4"]

    df_corrected = df.copy()

    epi_start_idx = (df_corrected["Time_s"] - epi_start).abs().idxmin()

    baseline_indices = range(epi_start_idx - baseline_points, epi_start_idx)

    for probe in probes:
        baseline = df_corrected.loc[baseline_indices, probe].mean()
        df_corrected[f"{probe}_corrected"] = df_corrected[probe] - baseline

    # Shift time axis
    shift = epi_start - baseline_points
    df_corrected["Time_s"] -= shift

    # Compute shifted timing
    epi_start_shifted = baseline_points
    epi_end_shifted = baseline_points + (epi_end - epi_start)

    return df_corrected, epi_start_shifted, epi_end_shifted


def step_jump_correction(df, probes=None, threshold=0.1):
    """
    Corrects sudden jumps in temperature traces.
    """
    if probes is None:
        probes = ["Probe_1", "Probe_2", "Probe_3", "Probe_4"]

    df_corrected = df.copy()
    for probe in probes:
        corrected = df_corrected[f"{probe}_corrected"].values.copy()
        diffs = np.diff(corrected)
        jump_idx = np.argmax(np.abs(diffs)) + 1
        offset = corrected[jump_idx] - corrected[jump_idx - 1]
        if abs(offset) > threshold:
            corrected[jump_idx:] -= offset
            print(f"[Step-Jump] {probe}: Jump at {jump_idx}, offset = {offset:.3f}°C")
        df_corrected[f"{probe}_corrected"] = corrected
    return df_corrected


def compute_temperature_metrics(
    df, probes=None, epi_start=0, epi_end=10, baseline_sds=None, threshold_factor=3
):

    if probes is None:
        probes = ["Probe_1", "Probe_2", "Probe_3", "Probe_4"]

    metrics = {}
    times = df["Time_s"].values

    for probe in probes:
        corrected = df[f"{probe}_corrected"].values

        # ---------------- threshold detection ----------------
        if baseline_sds is not None and probe in baseline_sds:
            threshold = threshold_factor * baseline_sds[probe]

            consecutive_required = 60
            above_thresh = corrected > threshold
            onset_idx = None

            for i in range(len(above_thresh) - consecutive_required + 1):
                if np.all(above_thresh[i : i + consecutive_required]):
                    onset_idx = i
                    break

            if onset_idx is not None:
                linear_start = times[onset_idx]
            else:
                print(
                    f"{probe}: threshold never exceeded for {consecutive_required} consecutive points. Skipping fit."
                )
                metrics[probe] = {"max_temp": np.nan, "slope_c_per_min": np.nan}
                continue

        else:
            linear_start = epi_start + 5

        epi_duration = epi_end - epi_start
        if epi_duration > 600:  # longer than 10 mins
            linear_end = epi_end
        else:
            linear_end = linear_start + 60

        slope_mask = (times >= linear_start) & (times <= linear_end)

        if np.sum(slope_mask) < 2:
            print(f"{probe}: Not enough points for regression. Skipping.")
            metrics[probe] = {"max_temp": np.nan, "slope_c_per_min": np.nan}
            continue

        # ---------------- slope fit ----------------
        result = linregress(times[slope_mask], corrected[slope_mask])
        print(f"{probe} R squared: {result.rvalue**2:.3f}")

        slope = result.slope * 60

        # ---------------- max ΔT ----------------
        max_idx = np.argmax(corrected)

        window = 5
        start = max(0, max_idx - window)
        end = min(len(corrected), max_idx + window + 1)

        max_temp = corrected[start:end].mean()

        metrics[probe] = {"max_temp": max_temp, "slope_c_per_min": slope}

    return metrics


def extract_metadata(file_path):
    filename = os.path.basename(file_path).replace(".txt", "").split("_")
    sequence_name = filename[3] if len(filename) > 3 else "Unknown"
    cable_length = filename[2] if len(filename) > 2 else "Unknown"
    config_number = "Unknown"
    for part in filename:
        if part.startswith("config"):
            config_number = part.replace("config", "")
            break
    return {
        "sequence_name": sequence_name,
        "cable_length": cable_length,
        "config_number": config_number,
        "config_label": f"Config {config_number}",
    }


def print_metrics_table(metrics, probes=None):
    if probes is None:
        probes = list(metrics.keys())

    data = []
    for probe in probes:
        max_temp = metrics[probe]["max_temp"]
        slope = metrics[probe]["slope_c_per_min"]
        max_temp_str = f"{max_temp:.2f}" if max_temp is not None else "N/A"
        slope_str = f"{slope:.2f}" if slope is not None else "N/A"
        data.append([probe, max_temp_str, slope_str])

    df_metrics = pd.DataFrame(data, columns=["Probe", "Max ΔT (°C)", "Slope (°C/min)"])
    print("\n--- Temperature Metrics ---")
    print(df_metrics.to_string(index=False))
    return df_metrics


def process_temperature_file(
    file, epi_start, epi_end, baseline_points=5, probes=None, baseline_sds=None
):
    df = parse_temperature_log(file)
    df, epi_start_shifted, epi_end_shifted = baseline_correct_temperature(
        df, epi_start, epi_end, baseline_points, probes
    )
    df = step_jump_correction(df, probes)
    metrics = compute_temperature_metrics(
        df, probes, epi_start_shifted, epi_end_shifted, baseline_sds=baseline_sds
    )
    metadata = extract_metadata(file)
    print_metrics_table(metrics)
    return df, metrics, metadata, epi_start_shifted, epi_end_shifted


def plot_temperature(
    df,
    metadata,
    probes=None,
    epi_start=0,
    epi_end=10,
    cooling_time=60,
    ax=None,
    title="",
):
    if probes is None:
        probes = ["Probe_1", "Probe_2", "Probe_3", "Probe_4"]

    styles = {"1": "arviz-viridish", "2": "arviz-plasmish", "3": "default"}
    style = styles.get(metadata["config_number"], "arviz-viridish")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    with az.style.context(style):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, probe in enumerate(probes):
            if metadata["config_number"] == "2":
                legend_label = probe_labels.get(probe, probe.replace("_", " ").title())
            else:
                legend_label = probe.replace("_", " ").title()

            ax.plot(
                df["Time_s"],
                df[f"{probe}_corrected"],
                label=legend_label,
                color=colors[i % len(colors)],
                marker="o",
                linestyle="",
                markersize=4,
                linewidth=2,
            )

        ax.axvspan(epi_start, epi_end, color="red", alpha=0.1)
        ax.set_xlim(0, cooling_time)
        ax.set_ylim(-2, 4)

        ax.set_xlabel("Time (s)", fontsize=20)
        ax.set_ylabel("ΔTemperature (°C)", fontsize=20)
        ax.set_title(title, fontsize=22)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(2.5)
        ax.spines["bottom"].set_linewidth(2.5)
        ax.tick_params(labelsize=18, width=2.5)

        legend_title = "Electrode" if metadata["config_number"] == "2" else "Probe"
        from matplotlib.patches import Patch

        epi_patch = Patch(facecolor="red", alpha=0.1, label="EPI")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(epi_patch)
        labels.append("EPI")

        ax.legend(
            handles,
            labels,
            title=legend_title,
            fontsize=14,
            title_fontsize=16,
            loc="upper left",
        )

    return ax


def plot_multiple_temperature_files(
    file_list,
    timing_info,
    plot_titles,
    figure_name,
    baseline_points=5,
    probes=None,
    shared_legend=True,
    baseline_sds=None,
):

    n_files = len(file_list)
    fig, axes = plt.subplots(1, n_files, figsize=(6 * n_files, 6), sharey=True)

    if n_files == 1:
        axes = [axes]

    dfs = []

    if shared_legend:
        all_handles, all_labels = [], []

    for i, file in enumerate(file_list):
        epi_start = timing_info["epi_start_times"][i]
        epi_end = timing_info["epi_stop_times"][i]
        shift = epi_start - baseline_points
        cooling = timing_info["cooling_times"][i] + epi_end - shift

        df, _metrics, metadata, epi_start_shifted, epi_end_shifted = (
            process_temperature_file(
                file, epi_start, epi_end, baseline_points, probes, baseline_sds
            )
        )
        dfs.append(df)

        ax = axes[i]
        plot_temperature(
            df,
            metadata,
            probes,
            epi_start_shifted,
            epi_end_shifted,
            cooling,
            ax=ax,
            title=plot_titles[i],
        )

        ax.text(
            -0.15,
            1.05,
            chr(65 + i) + ")",
            transform=ax.transAxes,
            fontsize=24,
            fontweight="bold",
        )

        if i != 0:
            ax.tick_params(labelleft=False)
            ax.set_ylabel("")

        if shared_legend:
            handles, labels = ax.get_legend_handles_labels()
            for h, l in zip(handles, labels):
                if l not in all_labels:
                    all_handles.append(h)
                    all_labels.append(l)
            ax.legend_.remove()

    if shared_legend:
        from matplotlib.patches import Patch

        epi_patch = Patch(facecolor="red", alpha=0.1)

        fig.subplots_adjust(left=0.1, bottom=0.25, top=0.95, right=0.95)

        fig.legend(
            all_handles + [epi_patch],
            all_labels + ["EPI"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            title="Legend",
            fontsize=18,
            title_fontsize=20,
            ncol=5,
            markerscale=2.5,
        )
    else:
        fig.subplots_adjust(left=0.1, bottom=0.1, top=0.95, right=0.95)

    fig.savefig(root / "figures" / f"{figure_name}.svg", bbox_inches="tight")
    fig.savefig(root / "figures" / f"{figure_name}.png", bbox_inches="tight")

    return dfs


### Establish Baseline SD ###
baseline_recording = parse_temperature_log(
    temperature_data / "100625_config1_10cm_baseline.txt"
)
probes = ["Probe_1", "Probe_2", "Probe_3", "Probe_4"]

baseline_std_dict = {}

for probe in probes:
    std = np.std(baseline_recording[probe])
    baseline_std_dict[probe] = std
    print(f"{probe} std: {std}")

### Plot Figure 04 ###
figure_04_data = [
    temperature_data / "100625_config3_10cm_shortest_1p5mmEPI_75s.txt",
    temperature_data / "100625_config3_10cm_shortest_2p5mmEPI_performance_75s.txt",
    temperature_data / "100625_config3_10cm_shortest_2p5mmEPI_normal_75s.txt",
]

timing_info = {
    "epi_start_times": [11, 11, 8],
    "epi_stop_times": [94, 94, 92],
    "cooling_times": [300, 300, 300],
}

plot_titles = [
    "1.5mm (GM: Performance)",
    "2.5mm (GM: Performance)",
    "2.5mm (GM: Normal)",
]
dfs_corrected = plot_multiple_temperature_files(
    figure_04_data,
    timing_info,
    plot_titles,
    "figure_04",
    baseline_sds=baseline_std_dict,
)

### Plot Figure 05 ###
figure_05_data = [
    temperature_data / "100625_config1_10cm_shortest_1p5mmEPI_75s.txt",
    temperature_data / "100625_config1_10cm_shortest_2p5mmEPI_performance_75s.txt",
    temperature_data / "100625_config1_10cm_shortest_2p5mmEPI_normal_75s.txt",
]

timing_info = {
    "epi_start_times": [12, 11, 8],
    "epi_stop_times": [94, 95, 92],
    "cooling_times": [300, 300, 300],
}

plot_titles = [
    "1.5mm (GM: Performance)",
    "2.5mm (GM: Performance)",
    "2.5mm (GM: Normal)",
]

dfs_corrected = plot_multiple_temperature_files(
    figure_05_data,
    timing_info,
    plot_titles,
    "figure_05",
    baseline_sds=baseline_std_dict,
)

### Plot Figure 06 ###

figure_06_data = [
    temperature_data / "100625_config1_10cm_1p5mmEPI_150s.txt",
    temperature_data / "100625_config1_10cm_2p5mmEPI_performance_150s.txt",
    temperature_data / "100625_config1_10cm_2p5mmEPI_normal_150s.txt",
]

timing_info = {
    "epi_start_times": [8, 11, 8],
    "epi_stop_times": [172, 170, 167],
    "cooling_times": [60, 60, 60],
}

plot_titles = [
    "1.5mm (GM: Performance)",
    "2.5mm (GM: Performance)",
    "2.5mm (GM: Normal)",
]
dfs_corrected = plot_multiple_temperature_files(
    figure_06_data,
    timing_info,
    plot_titles,
    "figure_06",
    baseline_sds=baseline_std_dict,
)

### Plot Figure 07 ###

figure_07_data = [
    temperature_data / "100625_config1_30cm_1p5mmEPI.txt",
    temperature_data / "100625_config1_30cm_2p5mmEPI_performance.txt",
]

timing_info = {
    "epi_start_times": [35, 9],
    "epi_stop_times": [115, 89],
    "cooling_times": [30, 30],
}
plot_titles = ["1.5mm (GM: Performance)", "2.5mm (GM: Performance)"]

dfs_corrected = plot_multiple_temperature_files(
    figure_07_data,
    timing_info,
    plot_titles,
    "figure_07",
    baseline_sds=baseline_std_dict,
)

### Plot Figure 08 ###

figure_08_data = [
    temperature_data / "100625_config1_30cm_1p5mmEPI_20min.txt",
    temperature_data / "100625_config2_30cm_1p5mmEPI_20min.txt",
]

# Timing (in s)
timing_info = {
    "epi_start_times": [10, 38],
    "epi_stop_times": [1218, 1246],
    "cooling_times": [600, 120],
}

plot_titles = ["1.5mm (GM: Performance)", "1.5mm (GM: Performance)"]

# Call the function to plot all
dfs_corrected = plot_multiple_temperature_files(
    figure_08_data,
    timing_info,
    plot_titles,
    "figure_08",
    shared_legend=False,
    baseline_sds=baseline_std_dict,
)
