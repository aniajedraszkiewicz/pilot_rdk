# ============================================================
# ENVIRONMENT SETUP (⚠️ Do not remove / keep at top-level)
# ============================================================
# PsychoPy reads some settings at import-time, so environment variables that affect
# backends must be set BEFORE importing psychopy.

import os, sys, re
os.environ["PSYCHOPY_USE_IOHUB"] = "True"   # use ioHub (reliable keyboard + timing)
os.environ["PSYCHOPY_NO_PTBOXLIB"] = "1"    # avoid PTB if it causes issues on your setup

# Make relative paths (e.g., results/) stable by running from this script's folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import psychopy
from psychopy import visual, monitors, core, gui, prefs
from psychopy.hardware import keyboard
from datetime import datetime
import numpy as np
import hashlib

import matplotlib
matplotlib.use("Agg")     # non-interactive backend (needed for saving PNGs without opening a window)
import matplotlib.pyplot as plt

# set PsychoPy prefs AFTER importing psychopy, BEFORE Window)
# winType controls the window backend
prefs.general['winType'] = 'pyglet'

# Request synchronization to vertical blank (improving timing stability)
prefs.general['waitBlanking'] = True

# Ensure keyboard uses ioHub (needed for kb.clock)
prefs.hardware['keyboard'] = 'iohub'

# Import experiment components
from .block import Block
from .rdk_stim import RDK


def plot_diagnostics(diagnostics, base_path):

    """
    Save a diagnostic plot showing the evolution of QUEST’s threshold estimate across trials.
    Use this to check whether the trajectory stabilizes and whether there are large jumps
    that might suggest unstable responding (or too few trials).
    """

    # Running threshold estimate after each QUEST update (one value per trial)
    thresh = diagnostics["threshold_estimates"]      
    fig = plt.figure(figsize=(7,4))
    trial_nums = np.arange(1, len(thresh) + 1)
    
    # Plot trajectory (dots + line) to see convergence across trials
    plt.plot(trial_nums, thresh, "o-", alpha=0.85)
    plt.xlabel("Trial")
    plt.ylabel("Threshold estimate (coherence)")
    plt.title("QUEST Threshold Trajectory")
    plt.grid(True)
    
    # Choose x-axis tick spacing so labels don't become cluttered in longer runs
    max_trials = len(thresh)
    step = 2 if max_trials <= 80 else 4   # auto choose spacing
    plt.xticks(np.arange(1, max_trials + 1, step))

    # Save and close to avoid accumulating figures across runs
    fig.savefig(base_path + "_thresholds.png")
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
        self.expInfo = {
        "participant": "",
        "screen_width_cm": "53.0",
        "viewing_distance_cm": "57.0",
        "fullscr": [True, False],
}

    
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
        raw = str(self.expInfo.get("participant", "")).strip() or "UNKNOWN"
        self.subject_id = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", raw).strip().strip(".") or "UNKNOWN"

        # Store fullscreen choice.
        self.fullscr = bool(self.expInfo.get("fullscr", True))

        # Prepare output location and a unique filename. Results are stored in ./results relative to the script directory
        os.makedirs("results", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_csv_path = os.path.join("results", f"{self.subject_id}_{stamp}.csv")
    
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
            color="black",
            waitBlanking=True,   # try to sync flips to the monitor refresh (vsync) for more stable frame timing
            useFBO=True,         # off-screen rendering; improves frame timing stability on some computers
        )

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
        self.dot_speed = 5.0               # deg/s 
        self.dot_density = 0.55            # dots/deg² 


        # Print a quick sanity check
        print(f"Measured refresh rate: {self.measured_rate:.2f} Hz")
    
    # Estimate the true refresh rate 
    def measure_refresh_rate(self):

        """
        1. Use PsychoPy getActualFrameRate() when available.
        2. Otherwise assume 60 Hz.
        3. Always record raw flip timing for later inspection 
        """
        # PsychoPy function that measures flip stability across frames and 
        # returns a refresh rate in Hz if timing is sufficiently stable
        refresh_rate_hz = self.win.getActualFrameRate(
            nIdentical=8,
            nMaxFrames=600,
            nWarmUpFrames=100,
            threshold=1.5,   # ms tolerance between consecutive frames
            infoMsg=None
        )
        
        if refresh_rate_hz is not None:
            refresh_rate_hz = float(refresh_rate_hz)
            self.refresh_rate_method = "getActualFrameRate"
        
        else:
            # PsychoPy could not determine a stable refresh rate
            refresh_rate_hz = 60.0
            self.refresh_rate_method = "default_60hz"
                      
        # Store flip-to-flip durations using the built-in function (seconds)
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
            print("Display timing is unreliable. Aborting experiment.")
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


    # Initialize ioHub keyboard and the RDK stimulus (including subject-specific RNG seed)
    # RDK RNG: deterministic seed from subject_id → controls dot randomness (signal/noise) for this participant.
    # Block RNG (direction): seeded inside Block.__init__() and logged later in run_experiment().
    def initialize_stimulus_and_load_trials(self):

        # Use ioHub for keyboard input (stable, stimulus-locked RTs); event-based polling can add extra latency.
        # Psychtoolbox/PTB keyboard backend is unavailable on this Mac setup, so ioHub is the reliable option here.
        self.kb = keyboard.Keyboard(backend='iohub')
        self.kb.clearEvents()
    
        # Create subject-specific RNG for the RDK dot stream 
        base = f"RDK|{self.subject_id}".encode("utf-8")
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

            # Timing / display
            f.write(self._section("DISPLAY TIMING"))

            # Single refresh rate used by the experiment
            f.write(self._line("Measured refresh rate used (Hz):", f"{self.measured_rate:.2f}"))

            f.write(self._line("Refresh rate estimation method:", getattr(self, "refresh_rate_method", "None")))

            # Flip-timing data collected during the refresh-rate check
            f.write(self._section("DISPLAY TIMING CHECK"))
            f.write(self._line("Refresh rate timestamps file:", os.path.basename(getattr(self, "frame_times_path", "None"))))
            f.write(self._line("Refresh rate intervals file:",os.path.basename(getattr(self, "frame_intervals_path", "None"))))
            
            # Stimulus parameters 
            f.write(self._section("RDK STIMULUS"))
            f.write(self._line("Dot speed (deg/s):", self.dot_speed))
            f.write(self._line("Dot density (dots/deg²/s):", self.dot_density))
            f.write(self._line("Total dots:", self.rdk.n_dots))
            f.write(self._line("Field diameter (deg):", self.rdk.field_diameter))

            # Per-sequence dot count 
            dots_per_seq = self.rdk.n_dots // self.rdk.n_sequences
            f.write(self._line("Dots per sequence:", dots_per_seq))

            # Environment: versions/backends that can affect timing and input 
            f.write(self._section("ENVIRONMENT"))
            f.write(self._line("PsychoPy version:", psychopy.__version__))
            f.write(self._line("Window backend:", prefs.general["winType"]))
            f.write(self._line("Keyboard backend:", prefs.hardware["keyboard"]))
            f.write(self._line("Screen width (cm):", self.screen_width_cm))
            f.write(self._line("Viewing distance (cm):", self.viewing_distance_cm))
            w, h = self.actual_win_size_px
            mw, mh = self.monitor_model_size_px
            f.write(self._line("Window framebuffer size(px):", f"{w} x {h}"))
            f.write(self._line("Monitor sizePix used for deg<->pix (px):", f"{mw} x {mh}"))


        print(f"\n Summary saved to: {os.path.abspath(summary_path)}")


    # Run the experiment block(s): show intro, run QUEST block, save diagnostics and plots
    # Pilot version: fixed trials are not loaded here. QUEST selects coherence adaptively inside Block.run_block().
    # Main version: replace QUEST with predetermined trials loaded from a CSV here.
    def run_experiment(self):
       
        # CSV header defines the order of columns written by Block.append_log_row()
        results_header = [
            "timestamp","subject_id",
            "block_no","trial_no","condition",
            "direction","coherence",
            "response_key","correct_key","is_correct",
            "reaction_time","timeout",
            "global_onset_time","response_flip_time","response_frame_idx",
            "response_detected_time",
            "stimulus_on_screen_duration",
            "frame_count","estimated_fps", "n_long_frames",
            "max_flip_interval" , "fix_onset_time","fix_offset_time","fix_duration","fix_target_sec",
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
            f.write(self._line("Block seed (direction RNG):", block.block_seed))

        # Show instructions before starting trials
        block.show_intro()  
    
        # Run adaptive QUEST block (trial loop + CSV logging happen inside Block.run_block())
        diagnostics = block.run_block()

        # Quick console feedback
        overall_accuracy = diagnostics["overall_accuracy"]
        print("Overall accuracy across QUEST trials:", round(overall_accuracy, 3))
       
        # Append QUEST diagnostics to the same summary file
        with open(summary_path, "a", encoding="utf-8") as f:
            
            # Keep the same section style as write_summary()
            f.write("\nQUEST DIAGNOSTICS\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'QUEST first intensity:':<34} {diagnostics['quest_first_intensity']:.4f}\n")
            f.write(f"{'QUEST intensity range:':<34} {diagnostics['quest_intensity_min']:.3f}–{diagnostics['quest_intensity_max']:.3f}\n")
            f.write(f"{'Final mean threshold:':<34} {diagnostics['mean']:.4f}\n")
            f.write(f"{'SD:':<34} {diagnostics['sd']:.4f}\n")
            f.write(f"{'Mode:':<34} {diagnostics['mode']:.4f}\n")
            f.write(f"{'Overall accuracy:':<34} {overall_accuracy:.3f}\n")
            ci_low, ci_high = diagnostics["ci_5_95"]
            f.write(f"{'CI 5–95%:':<34} ({ci_low:.2f}, {ci_high:.2f})\n")
            f.write(f"{'Last-10-trial accuracy:':<34} {diagnostics['last10_accuracy']:.3f}\n")
        
        # Save diagnostic plot using the same base name as the CSV
        diag_base = self.results_csv_path.replace(".csv", "")
        plot_diagnostics(diagnostics, diag_base)
        

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

   

