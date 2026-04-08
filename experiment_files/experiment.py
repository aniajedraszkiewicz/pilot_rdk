# ============================================================
# ENVIRONMENT SETUP (⚠️ Do not remove / keep at top-level)
# ============================================================
# PsychoPy reads some settings at import-time, so environment variables that affect
# backends must be set BEFORE importing psychopy.

import os, sys, re

# Preferred keyboard backend is passed in from the launcher file.
# "ptb" for PsychoPy Standalone
# "iohub" for running in editors like VS Code with a manually created virtual environment (venv)
PREFERRED_KEYBOARD_BACKEND = os.environ.get("RDK_KEYBOARD_BACKEND", "iohub")

if PREFERRED_KEYBOARD_BACKEND == "iohub":
    os.environ["PSYCHOPY_USE_IOHUB"] = "True"
    os.environ["PSYCHOPY_NO_PTBOXLIB"] = "1"
elif PREFERRED_KEYBOARD_BACKEND == "ptb":
    os.environ["PSYCHOPY_USE_IOHUB"] = "False"
    os.environ["PSYCHOPY_NO_PTBOXLIB"] = "0"
else:
    raise ValueError(f"Unsupported keyboard backend preference: {PREFERRED_KEYBOARD_BACKEND}")


# Make relative paths (e.g., results/) stable by running from this script's folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import psychopy
from psychopy import visual, monitors, core, gui, prefs, data
from psychopy.hardware import keyboard
from datetime import datetime
import numpy as np
import hashlib
import secrets

import matplotlib
matplotlib.use("Agg")     # non-interactive backend (needed for saving PNGs without opening a window)
import matplotlib.pyplot as plt

# set PsychoPy prefs AFTER importing psychopy, BEFORE Window)
# winType controls the window backend
prefs.general['winType'] = 'pyglet'

# Request synchronization to vertical blank (improving timing stability)
prefs.general['waitBlanking'] = True

# Choose preferred keyboard backend
prefs.hardware['keyboard'] = PREFERRED_KEYBOARD_BACKEND

# Refresh rate configuration: set this to your monitor's refresh rate (check System Settings → Displays).
# This value is used directly for stimulus timing; getActualFrameRate() below is only a sanity check — it does NOT override this.

HARDCODED_REFRESH_RATE_HZ = 120.0


# Import experiment components
from .block import Block
from .rdk_stim import RDK


def plot_diagnostics(diagnostics, base_path):
    """
    Plot QUEST diagnostics across trials.

    Panel:
        - Posterior threshold estimates in coherence (linear units)
        - Coherence values actually presented on each trial

    X-axis: trial number
    """
    # Extract data
    thresh_coh = diagnostics["threshold_estimates"]
    stim_coh = diagnostics.get("stimuli_used", None)

    # Number of trials completed in this QUEST block
    n = len(thresh_coh)
    trial_nums = np.arange(1, n + 1)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(
        trial_nums, thresh_coh,
        "o-", label="Threshold estimate (coherence)",
        linewidth=2
    )

    if stim_coh is not None and len(stim_coh) == n:
        ax.plot(
            trial_nums, stim_coh,
            "x", label="Stimulus shown (coherence)",
            alpha=0.8
        ) 

    ax.set_ylabel("Coherence (linear)")
    ax.set_xlabel("Trial")
    ax.set_title("QUEST diagnostics — coherence space")
    ax.grid(True)
    ax.legend(fontsize=8)

    y_max = max(1.0, np.nanmax(thresh_coh))
    if stim_coh is not None:
        y_max = max(y_max, np.nanmax(stim_coh))

    ax.set_ylim(0, y_max)
    ax.set_yticks(np.linspace(0, y_max, 6))
    # Final formatting
    step = 2 if n <= 80 else 4  # Reduce x-tick density for larger numbers of trials to keep the x-axis readable
    ax.set_xticks(np.arange(1, n + 1, step))

    fig.tight_layout()
    fig.savefig(base_path + "_quest_diagnostics.png")
    plt.close(fig)

def fit_and_plot_psychometric(accuracy_by_level, summary_path, section_fn, line_fn, base_path):
    """
    Fit a Weibull psychometric function to validation accuracy data,
    write results to the summary file, and save a plot.
    Uses PsychoPy built-in data.FitWeibull.
    """
    coherences = sorted(accuracy_by_level.keys())
    proportions = [accuracy_by_level[c] for c in coherences]
 
    fit = data.FitWeibull(coherences, proportions, expectedMin=0.5, display=0)
    threshold = float(fit.inverse(0.75))
 
    # Write fit results to summary file
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(section_fn("VALIDATION — PSYCHOMETRIC FIT (Weibull)"))
        f.write(line_fn("Threshold (75% correct):", round(threshold, 4)))
        f.write(line_fn("Slope:", round(float(fit.params[1]), 4)))
        for coh in coherences:
            val = accuracy_by_level[coh]
            f.write(line_fn(f"  Accuracy at coherence {coh}:", round(val, 3) if val is not None else "NA"))
 
    # Plot data points and fitted curve
    x_fit = np.linspace(min(coherences) * 0.5, max(coherences) * 1.2, 200)
    y_fit = [float(fit.eval(x)) for x in x_fit]
 
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(coherences, proportions, "o", markersize=8, label="Validation data")
    ax.plot(x_fit, y_fit, "-", linewidth=2, label="Weibull fit")
    ax.axvline(threshold, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.75, color="gray", linestyle="--", linewidth=1)
    ax.annotate(f"threshold = {threshold:.3f}",
                xy=(threshold, 0.75), xytext=(threshold + 0.02, 0.65),
                fontsize=9, color="gray")
    ax.set_xlabel("Coherence")
    ax.set_ylabel("Proportion correct")
    ax.set_title("Psychometric function — validation block")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(base_path + "_psychometric.png")
    plt.close(fig)
 
 
def plot_accuracy_comparison(practice_07, practice_04, validation, base_path):
    """
    Bar chart comparing accuracy across practice blocks and each validation coherence level.
    """
    # Build labels and values
    labels = ["Practice 0.7", "Practice 0.4"]
    values = [practice_07["final_accuracy"], practice_04["final_accuracy"]]
    colors = ["steelblue", "darkorange"]
 
    acc = validation["accuracy_by_level"]
    for coh in sorted(acc.keys()):
        labels.append(f"Validation {coh}")
        values.append(acc[coh] if acc[coh] is not None else 0.0)
        colors.append("seagreen")
 
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
 
    ax.axhline(0.75, color="gray", linestyle="--", linewidth=1, label="75% criterion")
    ax.axhline(0.5, color="lightgray", linestyle=":", linewidth=1, label="Chance (50%)")
 
    # Label each bar with its value
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
 
    ax.set_ylabel("Proportion correct")
    ax.set_title("Accuracy by block and coherence level")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.grid(axis="y")
 
    fig.tight_layout()
    fig.savefig(base_path + "_accuracy_comparison.png")
    plt.close(fig)
 
class Experiment:
    """
    This class is a high-level experiment controller. It is responsible for:
    - collecting basic info about the participant (e.g., ID) 
    - creating the PsychoPy Monitor/Window (units='deg' depend on correct monitor geometry)
    - measuring the refresh rate
    - defining stimulus parameters (e.g., dot speed, dot density)
    - enabling/disabling debug output for timing and frame-by-frame diagnostics
    - running the experiment block(s)
    - saving trial-level data and run metadata (e.g., summary file, diagnostic plots) 
    """
    
    # Define the fields shown in the startup GUI dialog 
    def __init__(self):
        self.keyboard_backend_preferred = PREFERRED_KEYBOARD_BACKEND
        self.expInfo = {
        "participant": "",
        "screen_width_cm": "47.2",
        "viewing_distance_cm": "57.0",
        "fullscr": [True, False],}

    
    # Run all preparation steps in a fixed order (GUI → Window → calibration → stimulus)
    def setup(self):
        self.collect_participant_info()
        self.create_window_and_monitor()
        self.measure_and_define_parameters()
        self.initialize_stimulus_and_load_trials()

    # Collect participant ID and create output file paths
    def collect_participant_info(self):
        print(
            "\n[NOTE] Fullscreen is required for accurate visual-angle (deg) calibration. "
            "Windowed mode should be used for debugging only.\n"
        )
       
        # Show GUI; user inputs are written back into self.expInfo
        dlg = gui.DlgFromDict(self.expInfo, title='RDK Display Settings')
        if not dlg.OK:
            core.quit()

        # Store choices from the dialog. This removes characters forbidden by Windows filesystems
        # and strips trailing dots or spaces, which can also cause file-saving errors.
        raw = str(self.expInfo.get("participant", "")).strip()

        # If no ID provided, use a simple test label (NOT unique)
        self.subject_id = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", raw).strip().strip(".")
        if self.subject_id == "":
            self.subject_id = "TEST"

        # Store fullscreen choice.
        self.fullscr = bool(self.expInfo.get("fullscr", True))

        # Prepare output location and a unique run identifier
        os.makedirs("results", exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)

        # Short, clean filename (no subject ID duplication)
        self.results_csv_path = os.path.join("results", f"{self.run_id}.csv")
    
    # Create a PsychoPy Monitor (geometry container) and the PsychoPy Window
    def create_window_and_monitor(self):
        # Parse physical geometry (needed for units="deg": degrees ↔ cm ↔ pixels)
        try:
            screen_width_cm = float(str(self.expInfo["screen_width_cm"]).replace(",", "."))
            viewing_distance_cm = float(str(self.expInfo["viewing_distance_cm"]).replace(",", "."))
        except Exception:
            print("ERROR: Please enter valid screen width and viewing distance (e.g., 53.0, 60.0).")
            core.quit()

        # Build Monitor object (geometry model used for deg↔pix conversion). 
        # Avoid writing to the global PsychoPy Monitor Center (mon.save()).
        # Writing monitor profiles can create cross-machine drift and persistent side effects.
        mon = monitors.Monitor("CurrentDisplay")
        mon.setWidth(screen_width_cm)            # cm
        mon.setDistance(viewing_distance_cm)     # cm

        # Placeholder sizePix required before Window creation (units="deg"). Replaced with the true screen sizePix only in fullscreen.
        mon.setSizePix((800, 600))               
           
        self.mon = mon
           
        # Create the PsychoPy Window; now the real pixel size becomes available via win.size
        self.win = visual.Window(
            monitor=mon,
            units="deg",
            fullscr=self.fullscr,
            color=[-1, -1, -1],
            colorSpace='rgb',
            waitBlanking=True,   # try to sync flips to the monitor refresh (vsync) for more stable frame timing
            useFBO=True,         # off-screen rendering; improves frame timing stability on some computers
        )
        self.win.mouseVisible = False

        # Debug: Retina Macs often report win.size in framebuffer pixels (2× scale)
        print("win.useRetina =", getattr(self.win, "useRetina", None))
        print("win.size =", self.win.size)
      
        # win_px = "window pixels as reported by PsychoPy" (often framebuffer px on Retina)
        win_px = (int(self.win.size[0]), int(self.win.size[1]))
        
        # model_px = "pixel size intended to use for deg↔pix geometry" (i.e., what mon.sizePix should become in fullscreen).
        # Default: assume win.size already matches the screen geometry
        model_px = win_px

        # macOS Retina: convert framebuffer px -> logical px for deg↔pix geometry
        if sys.platform == "darwin" and getattr(self.win, "useRetina", False):
            model_px = (win_px[0] // 2, win_px[1] // 2)

        # Read current Monitor sizePix (often still the placeholder)
        before_px = tuple(mon.getSizePix())

        # Only update Monitor sizePix in fullscreen; in fullscreen, win.size corresponds to the physical display (after Retina adjustment),
        # so it’s valid for deg geometry (windowed mode cannot provide valid screen resolution for deg calibration)
        if self.fullscr:
            if before_px != model_px:
                mon.setSizePix(model_px)
        else:
            print("[INFO] Windowed mode: not overwriting mon.sizePix from win.size (would break deg calibration).")

        # Log what happened (reproducibility/debugging)
        after_px = tuple(mon.getSizePix())
        print(f"[WINDOW] win.size(px)={win_px}, mon.sizePix(before)={before_px}, mon.sizePix(after)={after_px}, model_px={model_px}")

        # Store for summary/logging
        self.screen_width_cm = float(mon.getWidth())
        self.viewing_distance_cm = float(mon.getDistance())
        self.actual_win_size_px = win_px
        self.monitor_model_size_px = model_px
   
    # Measure refresh rate and define parameters
    def measure_and_define_parameters(self):
        self.measured_rate = self.measure_refresh_rate()
        target_displacement = 0.28                                       # deg — MT Dopt range (Pilly and Seitz, 2009)
        self.dot_speed = target_displacement * self.measured_rate / 3.0  # auto-corrects for any Hz
        self.dot_density = 24.0                                          # dots/deg²/s 

        # Print a quick sanity check
        print(f"Measured refresh rate: {self.measured_rate:.2f} Hz")
    
    # Estimate the true refresh rate 
    def measure_refresh_rate(self):

        """
        Uses HARDCODED_REFRESH_RATE_HZ as the authoritative refresh rate.
        getActualFrameRate() runs as a sanity check only — it logs a warning
        if the measured rate differs too much, but never overrides the hardcoded value.
        """
        # Sanity check (does NOT set the rate used by the experiment)

        measured= self.win.getActualFrameRate(
            nIdentical=8,
            nMaxFrames=600,
            nWarmUpFrames=100,
            threshold=1.5,   # ms tolerance between consecutive frames
            infoMsg=None
        )
        
        if measured is None:
            print("[WARNING] getActualFrameRate() could not measure a stable rate. "
              "Proceeding with hardcoded value. Check vsync and display settings.")
        
        else:
            diff = abs(measured - HARDCODED_REFRESH_RATE_HZ)
            if diff > 2.0:
                print(f"[WARNING] getActualFrameRate() returned {measured:.2f} Hz, "
                    f"but hardcoded rate is {HARDCODED_REFRESH_RATE_HZ} Hz "
                    f"(difference: {diff:.2f} Hz). "
                    f"Check System Settings → Displays before running the experiment.")
            else:
                print(f"[OK] getActualFrameRate() returned {measured:.2f} Hz — "
                    f"consistent with hardcoded {HARDCODED_REFRESH_RATE_HZ} Hz.")
                      
        # Use hardcoded rate for all stimulus timing 
        refresh_rate_hz = HARDCODED_REFRESH_RATE_HZ
        self.refresh_rate_method = "hardcoded"

        # Record flip timing for later inspection. Store flip-to-flip durations using the built-in function (seconds)
        self.win.recordFrameIntervals = True
        self.win.frameIntervals = []  # clear any previous data

        # Store flip timestamps returned by win.flip() (manual timing trace as an extra check)
        frame_times = []
        for _ in range(240):
            frame_times.append(self.win.flip())

        # Stop recording frame intervals (record only during refresh-rate check)
        self.win.recordFrameIntervals = False

        # Safety check (need at least 2 timestamps to form at least 1 interval)
        if len(frame_times) <= 1:
            print("[ERROR] Flip timing failed (not enough timestamps).")
            self.win.close()
            core.quit()

        # Store timing data
        self.refresh_frame_times = frame_times                      # timestamps from win.flip()
        self.refresh_frame_intervals = list(self.win.frameIntervals)  # PsychoPy recorded intervals

        # Save timing data in additional files
        base = self.results_csv_path.replace(".csv", "")
        self.frame_times_path = base + "_refresh_timestamps.txt"
        self.frame_intervals_path = base + "_refresh_intervals.txt"

        np.savetxt(self.frame_times_path, self.refresh_frame_times)
        np.savetxt(self.frame_intervals_path, self.refresh_frame_intervals)

        print(f"Refresh rate USED: {refresh_rate_hz:.2f} Hz ({self.refresh_rate_method})")
        return float(refresh_rate_hz)


    # Initialize keyboard backend and the RDK stimulus (including subject-specific RNG seed)
    # RDK RNG: deterministic seed from subject_id → controls dot randomness (signal/noise) for this participant.
    # Block RNG (direction): seeded inside Block.__init__() and logged later in run_experiment().
    def initialize_stimulus_and_load_trials(self):

        # Initialize keyboard backend
        self.kb = keyboard.Keyboard(backend=self.keyboard_backend_preferred)
        self.keyboard_backend_used = self.keyboard_backend_preferred
        self.kb.clearEvents()

        print(f"[INFO] Keyboard backend: {self.keyboard_backend_used}")
    
        # Create subject-specific RNG for the RDK dot stream 
        base = f"RDK|{self.subject_id}|{self.run_id}".encode("utf-8")
        digest = hashlib.sha256(base).hexdigest()
        seed_rdk = int(digest[:8], 16)  # first 8 hex digits -> 32-bit int
        self.rdk_seed = seed_rdk
        rdk_rng = np.random.default_rng(seed_rdk)

        # Initialize RDK stimulus using the measured refresh rate 
        self.rdk = RDK(
            self.win,
            frame_rate=self.measured_rate,      # measured refresh rate (Hz)
            dot_speed=self.dot_speed,            
            dot_density=self.dot_density,   
            rng=rdk_rng    
        )
        print("n_dots:", self.rdk.n_dots)
        print("field_area:", np.pi * (self.rdk.field_radius ** 2))
        print("measured_rate:", self.measured_rate)
        print("dot_density:", self.dot_density)


        # Write a metadata summary file 
        self.write_summary()
 
    # Create formatting helpers (shared by write_summary and run_experiment) 
    def _section(self, title, width=60):
        return f"\n{title}\n" + ("=" * width) + "\n"

    def _line(self, label, value, label_width=34):
        return f"{label:<{label_width}} {value}\n"
    
    # Write a summary text file with key parameters and environment metadata
    def write_summary(self):
        summary_path = self.results_csv_path.replace(".csv", "_summary.txt")
        
        with open(summary_path, "w", encoding="utf-8") as f:
            # Header 
            f.write("RUN SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(self._line("Participant:", self.subject_id))
            f.write(self._line("Timestamp:", datetime.now().isoformat(timespec="seconds")))
            f.write(self._line("Run ID:", self.run_id))

            # Timing / display
            f.write(self._section("DISPLAY TIMING"))

            # Single refresh rate used by the experiment
            f.write(self._line("Hardcoded refresh rate used (Hz):", f"{self.measured_rate:.2f}"))

            f.write(self._line("Refresh rate estimation method:", getattr(self, "refresh_rate_method", "None")))

            # Flip-timing data collected during the refresh-rate check
            f.write(self._section("DISPLAY TIMING CHECK"))
            f.write(self._line("Refresh rate timestamps file:", os.path.basename(getattr(self, "frame_times_path", "None"))))
            f.write(self._line("Refresh rate intervals file:",os.path.basename(getattr(self, "frame_intervals_path", "None"))))
            
            # Stimulus parameters 
            f.write(self._section("RDK STIMULUS"))
            f.write(self._line("Dot speed (deg/s):", round(self.dot_speed,2)))
            f.write(self._line("Dot density (dots/deg²/s):", self.dot_density))
            f.write(self._line("Total dots:", self.rdk.n_dots))
            f.write(self._line("Spatial displacement (deg):", round(self.rdk.spatial_displacement, 2)))
            f.write(self._line("Temporal displacement (ms):", round(self.rdk.temporal_displacement, 2)))
            f.write(self._line("Instantaneous dot density (dots/deg²):", round(self.rdk.instantaneous_dot_density, 2)))
            f.write(self._line("Field diameter (deg):", self.rdk.field_diameter))
            f.write(self._line("Field area (deg²):", round(np.pi * (self.rdk.field_radius ** 2),2)))


            # Per-sequence dot count 
            dots_per_seq = self.rdk.n_dots // self.rdk.n_sequences
            f.write(self._line("Dots per sequence:", dots_per_seq))

            # Environment: versions/backends that can affect timing and input 
            f.write(self._section("ENVIRONMENT"))
            f.write(self._line("PsychoPy version:", psychopy.__version__))
            f.write(self._line("Window backend:", prefs.general["winType"]))
            f.write(self._line("Keyboard backend preferred:", prefs.hardware["keyboard"]))
            f.write(self._line("Keyboard backend used:", self.keyboard_backend_used))
            f.write(self._line("Screen width (cm):", self.screen_width_cm))
            f.write(self._line("Viewing distance (cm):", self.viewing_distance_cm))
            w, h = self.actual_win_size_px
            mw, mh = self.monitor_model_size_px
            f.write(self._line("Window framebuffer size(px):", f"{w} x {h}"))
            f.write(self._line("Monitor sizePix used for deg<->pix (px):", f"{mw} x {mh}"))


        print(f"\n Summary saved to: {os.path.abspath(summary_path)}")


    # Run the adaptive pilot experiment: practice (0.7), practice (0.4), validation, QUEST.

    def run_experiment(self):
       
        # CSV header defines the order of columns written by Block.append_log_row()
        results_header = [
            "timestamp","subject_id",
            "block_no","trial_no","condition",
            "direction","coherence", "intensity_log10",
            "threshold_estimate_log10",
            "threshold_estimate_coh",
            "response_key","correct_key","is_correct",
            "reaction_time","timeout",
            "global_onset_time","response_flip_time","response_frame_idx",
            "response_detected_time",
            "stimulus_on_screen_duration",
            "frame_count","estimated_fps", "n_long_frames",
            "max_flip_interval" ,
            "fix_onset_time","fix_offset_time","fix_duration","fix_target_sec",
        ]
        
        # Single block for now (extend to multiple blocks later if needed)
        block_no = 1
         
        # Create an instance of the Block class for this run.
        # - Uses already-initialized objects from Experiment: Window, keyboard, and RDK stimulus.
        # - The block writes trial rows to results_csv_path using results_header.
        block = Block(
            self.win,
            self.kb,
            self.rdk,
            block_no,
            self.subject_id,
            self.results_csv_path,
            results_header,
            max_stim_sec=15.0,
            debug=False             # Change debug only here; keep Block/Trial code unchanged
        )


        # Path to the run summary text file created in write_summary()
        summary_path = self.results_csv_path.replace(".csv", "_summary.txt")
 
        
        # Append block-level RNG seed (created inside Block.__init__()).
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(self._section("RNG SEEDS"))
            f.write(self._line("RDK seed (dot-level randomness):", self.rdk_seed))
            f.write(self._line("RDK seed base:", f"RDK|{self.subject_id}|{self.run_id}"))
            f.write(self._line("Block seed (direction RNG):", block.block_seed))

        # Show instructions before starting trials
        block.show_intro()  

        # Run the adaptive pilot: practice (0.7), practice (0.4), validation, QUEST
        pilot_results = block.run_adaptive_pilot()
        diagnostics = pilot_results["quest"] if pilot_results else None
 
        # If the block was aborted (ESC), stop cleanly
        if diagnostics is None:
            return
 
        # Show outro to participant (break / end message)
        block.show_outro()

        # Base path for all output files (plots, etc.)
        diag_base = self.results_csv_path.replace(".csv", "")

        # Quick console feedback
        overall_accuracy = diagnostics["overall_accuracy"]
        print("Overall accuracy across QUEST trials:", round(overall_accuracy, 3))
       
        # Append practice and validation results to the summary file
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(self._section("PRACTICE (coherence 0.7)"))
            f.write(self._line("Trials completed:", pilot_results["practice_07"]["n_trials"]))
            f.write(self._line("Final accuracy:", round(pilot_results["practice_07"]["final_accuracy"], 3)))
 
            f.write(self._section("PRACTICE (coherence 0.4)"))
            f.write(self._line("Trials completed:", pilot_results["practice_04"]["n_trials"]))
            f.write(self._line("Final accuracy:", round(pilot_results["practice_04"]["final_accuracy"], 3)))
 
            f.write(self._section("VALIDATION"))
            f.write(self._line("Trials completed:", pilot_results["validation"]["n_trials"]))
            f.write(self._line("Final accuracy:", round(pilot_results["validation"]["final_accuracy"], 3)))
        
        
         # Fit Weibull, write results to summary, and save plot
        try:
            fit_and_plot_psychometric(
                pilot_results["validation"]["accuracy_by_level"],
                summary_path, self._section, self._line, diag_base
            )
        except Exception as e:
            print(f"[WARNING] Psychometric fit/plot failed: {e}")

        # Append QUEST diagnostics to the same summary file
        with open(summary_path, "a", encoding="utf-8") as f:
            
            # Keep the same section style as write_summary()
            f.write("\nQUEST DIAGNOSTICS\n")
            f.write("=" * 60 + "\n")

            # Helper: format numeric values for text output - returns a string with fixed decimal precision 
            # or "NA" if the value is missing. Used to keep the summary readable and prevent crashes on None values.
            def _fmt(x, nd=4):
                return "NA" if x is None else f"{float(x):.{nd}f}"

            # Print key QUEST settings 
            f.write("QUEST SETTINGS\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Prior threshold guess (coh):':<34} {_fmt(diagnostics.get('quest_start_coh'), 2)}\n")
            f.write(f"{'Minimum coherence:':<34} {_fmt(diagnostics.get('quest_min_coh'), 2)}\n")
            f.write(f"{'Maximum coherence:':<34} {_fmt(diagnostics.get('quest_max_coh'), 2)}\n")
            f.write(f"{'Prior threshold guess (log10):':<34} {_fmt(diagnostics.get('quest_start_intensity_log10'), 2)}\n")
            f.write(f"{'Prior SD (log10):':<34} {_fmt(diagnostics.get('quest_start_intensity_sd_log10'), 2)}\n")
            f.write(f"{'Target accuracy:':<34} {_fmt(diagnostics.get('quest_pThreshold'), 2)}\n")
            f.write(f"{'Guess rate (gamma):':<34} {_fmt(diagnostics.get('quest_gamma'), 2)}\n")
            f.write(f"{'Slope (beta):':<34} {_fmt(diagnostics.get('quest_beta'), 2)}\n")
            f.write(f"{'Lapse rate (delta):':<34} {_fmt(diagnostics.get('quest_delta'), 2)}\n")
            f.write(f"{'Number of trials:':<34} {diagnostics.get('quest_nTrials', 'NA')}\n")
            f.write(f"{'Threshold estimation rule:':<34} {diagnostics.get('quest_method', 'NA')}\n")

            f.write("\nQUEST RESULTS\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Coherence range:':<34} "f"{_fmt(diagnostics.get('coh_min_used'), 2)}–{_fmt(diagnostics.get('coh_max_used'), 2)}\n")
            f.write(f"{'Final threshold estimate (coh):':<34} {_fmt(diagnostics.get('mean_coh'), 2)}\n")
            f.write(f"{'Overall accuracy:':<34} {_fmt(diagnostics.get('overall_accuracy'), 2)}\n")
            # confidence intervals for coherence
            ci = diagnostics.get("ci_5_95_coh", None)
            if ci is None or len(ci) != 2:
                f.write(f"{'CI 5–95% (coh):':<34} (NA, NA)\n")
            else:
                ci_low, ci_high = ci
                f.write(f"{'CI 5–95% (coh):':<34} ({_fmt(ci_low, 2)}, {_fmt(ci_high, 2)})\n")
            # confidence intervals for intensity (coherence in log10)
            ci_log10 = diagnostics.get("ci_5_95_log10", None)
            if ci_log10 is None or len(ci_log10) != 2:
                f.write(f"{'CI 5–95% (log10):':<34} (NA, NA)\n")
            else:
                ci_low_log10, ci_high_log10 = ci_log10
                f.write(f"{'CI 5–95% (log10):':<34} ({_fmt(ci_low_log10, 2)}, {_fmt(ci_high_log10, 2)})\n")
            f.write(f"{'Last-10-trial accuracy:':<34} {_fmt(diagnostics.get('last10_accuracy'), 2)}\n")
            f.write("\nRESPONSE BIAS (POST-HOC)\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'Sensitivity':<34} "
                    f"{_fmt(diagnostics.get('bias_sensitivity_right'), 2)}\n")
            f.write(f"{'Specificity':<34} "
                    f"{_fmt(diagnostics.get('bias_specificity_right'), 2)}\n")

        
        # Save QUEST diagnostic plot
        plot_diagnostics(diagnostics, diag_base)
 
        # Save accuracy comparison across all blocks and coherence levels
        try:
            plot_accuracy_comparison(pilot_results["practice_07"], pilot_results["practice_04"], pilot_results["validation"], diag_base)
        except Exception as e:
            print(f"[WARNING] Accuracy comparison plot failed: {e}")
        

    # Close the PsychoPy window and quit
    def close_exp(self):
        self.win.close()
        core.quit()


# Run the experiment only when this file is executed directly (not when imported as a module)
if __name__ == "__main__":
    
    # create the top-level experiment controller
    exp = Experiment()
    
    # prepare GUI/window/calibration/stimulus
    exp.setup()
    
    # run block(s) and save outputs
    exp.run_experiment()

    # close window and quit
    exp.close_exp()

   

