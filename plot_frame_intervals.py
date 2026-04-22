"""
Plot frame intervals saved during the experiment.

BEFORE RUNNING: edit the settings below.

Run from your project folder:
    python plot_frame_intervals.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# EDIT THESE BEFORE RUNNING

REFRESH_HZ = 120  # your monitor refresh rate in Hz

RESULTS_DIR = "/Users/annajedraszkiewicz/dokumenty/programming_projects/pilot_rdk/experiment_files/results"

# list the exact filenames you want to plot (just the filenames, not full paths)
FILES = [
    "20260422_151817_191c_frame_intervals_practice_70.txt",
    "20260422_151817_191c_frame_intervals_practice_40.txt",
    "20260422_151817_191c_frame_intervals_validation.txt",
    "200260422_151817_191c_frame_intervals_quest.txt",
]

# ============================================================

EXPECTED_MS       = 1000.0 / REFRESH_HZ
LONG_THRESHOLD_MS = EXPECTED_MS * 1.5

print(f"Refresh rate : {REFRESH_HZ} Hz")
print(f"Expected     : {EXPECTED_MS:.2f} ms per frame")
print(f"Long frame   : > {LONG_THRESHOLD_MS:.2f} ms")

# load whichever files exist and are non-empty
loaded = {}
for filename in FILES:
    path = os.path.join(RESULTS_DIR, filename)
    label = filename.replace(".txt", "").split("_frame_intervals_")[-1]  # e.g. "practice_70"
    if not os.path.exists(path):
        print(f"Not found, skipping : {path}")
        continue
    try:
        data = np.loadtxt(path)
        if data.size == 0:
            print(f"Empty file, skipping: {path}")
            continue
        loaded[label] = data * 1000  # seconds -> ms
        print(f"Loaded : {filename}  ({len(data)} frames)")
    except Exception as e:
        print(f"Could not read, skipping: {filename} ({e})")

if not loaded:
    print("\nNo valid files found. Check FILES and RESULTS_DIR.")
    raise SystemExit(1)

# one row of plots per file, two columns: histogram + time series
n = len(loaded)
fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
if n == 1:
    axes = [axes]

for ax_row, (label, intervals) in zip(axes, loaded.items()):

    n_frames = len(intervals)
    n_long   = int(np.sum(intervals > LONG_THRESHOLD_MS))
    mean_ms  = float(np.mean(intervals))
    max_ms   = float(np.max(intervals))

    print(f"\n--- {label} ---")
    print(f"  frames      : {n_frames}")
    print(f"  mean        : {mean_ms:.2f} ms  (expected {EXPECTED_MS:.2f} ms)")
    print(f"  max         : {max_ms:.2f} ms")
    print(f"  long frames : {n_long} / {n_frames}  ({100*n_long/n_frames:.1f}%)")

    # histogram
    ax_hist = ax_row[0]
    ax_hist.hist(intervals, bins=100, color="steelblue", edgecolor="none")
    ax_hist.axvline(EXPECTED_MS,       color="green", linewidth=1.5, linestyle="--", label=f"expected ({EXPECTED_MS:.1f} ms)")
    ax_hist.axvline(LONG_THRESHOLD_MS, color="red",   linewidth=1.5, linestyle="--", label=f"long frame ({LONG_THRESHOLD_MS:.1f} ms)")
    ax_hist.set_xlabel("Frame interval (ms)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"{label} — histogram\nmean={mean_ms:.2f} ms  |  long frames={n_long} ({100*n_long/n_frames:.1f}%)")
    ax_hist.legend()

    # time series
    ax_time = ax_row[1]
    ax_time.plot(intervals, color="steelblue", linewidth=0.5, alpha=0.7)
    ax_time.axhline(EXPECTED_MS,       color="green", linewidth=1.5, linestyle="--", label=f"expected ({EXPECTED_MS:.1f} ms)")
    ax_time.axhline(LONG_THRESHOLD_MS, color="red",   linewidth=1.5, linestyle="--", label=f"long frame ({LONG_THRESHOLD_MS:.1f} ms)")
    ax_time.set_xlabel("Frame number")
    ax_time.set_ylabel("Interval (ms)")
    ax_time.set_title(f"{label} — over time")
    ax_time.legend()

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "frame_intervals_plot.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to: {out_path}")
plt.show()