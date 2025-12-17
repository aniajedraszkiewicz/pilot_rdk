# ============================================================
# ENVIRONMENT SETUP (⚠️ Do not remove / keep at top-level)
# ============================================================
# PsychoPy reads some settings at import-time, so environment variables that affect
# backends must be set BEFORE importing psychopy.

import os, sys
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
    - collecting basic info about the participant (e.g., ID) and monitor choice
    - creating the PsychoPy Monitor/Window (units='deg' depend on correct monitor geometry)
    - stabilizing display timing before calibration (warm-up flips) and measuring the refresh rate
    - defining stimulus parameters (e.g., dot speed, dot density) and applying refresh-rate corrections when needed
    - enabling/disabling debug output for timing and frame-by-frame diagnostics
    - running the experiment block(s)
    - saving trial-level data and run metadata (e.g., summary file, diagnostic plots) 
    """
    
    # Define the fields shown in the startup GUI dialog 
    def __init__(self):
        self.expInfo = {
        'participant': '',
        'monitor': ['MacBookDisplay', 'OtherDisplay']
    }

    # Run all preparation steps in a fixed order (GUI → Window → calibration → stimulus)
    def setup(self):
        self.collect_participant_info()
        self.create_window_and_monitor()
        self.measure_and_define_parameters()
        self.initialize_stimulus_and_load_trials()

    # Collect participant ID + monitor choice and create output file paths
    def collect_participant_info(self):
       
        # Show GUI; user inputs are written back into self.expInfo
        dlg = gui.DlgFromDict(self.expInfo, title='RDK Display Settings')
        if not dlg.OK:
            core.quit()

        # Store choices from the dialog
        self.monitor_choice = self.expInfo['monitor']
        self.subject_id = self.expInfo['participant'].strip() or "UNKNOWN"

        # Prepare output location and a unique filename. Results are stored in ./results relative to the script directory
        os.makedirs("results", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_csv_path = os.path.join("results", f"{self.subject_id}_{stamp}.csv")
    
    # Create a monitor profile and the PsychoPy Window 
    def create_window_and_monitor(self):
        if self.monitor_choice == "MacBookDisplay":
            # Create the MacBookDisplay Monitor profile if it doesn't exist yet.
            # This stores screen geometry for deg↔pix conversion when units='deg')
            if "MacBookDisplay" not in monitors.getAllMonitors():
                mon = monitors.Monitor("MacBookDisplay")
                mon.setWidth(30.41)          # cm 
                mon.setDistance(60.0)        # cm 
                mon.setSizePix((1440, 900))  # macOS logical resolution
                mon.save()
            
            # Load the saved profile
            mon = monitors.Monitor("MacBookDisplay")
            print("Using monitor profile: MacBookDisplay")

        else:
            # Default monitor profile used when running on a non-MacBook display
            mon = monitors.Monitor('OtherDisplay')
            mon.setWidth(53.0)
            mon.setDistance(60.0)
            mon.setSizePix((1920, 1080))
            print("Using OtherDisplay profile (safe defaults).")

        # Create the PsychoPy Window (units='deg' requires a valid Monitor geometry)
        self.win = visual.Window(
            monitor=mon,
            size=mon.getSizePix(), # window resolution (should match the monitor profile)
            units='deg',
            color='black',
            fullscr=False,
            waitBlanking=True,   # try to sync flips to the monitor refresh (vsync) for more stable frame timing
            useFBO=True,         # off-screen rendering; improves frame timing stability on this MacBook
        )

        # Warm up display timing before any measurements.
        # The first few screen updates can be unstable, so we flip the window
        # several times to let timing settle before measuring refresh rate
        print("\n Warming up display timing (120 flips)...")
        for _ in range(120):
            self.win.flip()
        print(" Warm-up complete.")
   
    # Measure refresh rate and compute derived stimulus parameters (density correction, sanity checks)
    def measure_and_define_parameters(self):
        self.measured_rate = self.measure_refresh_rate()

        # Define design constants
        self.DESIGN_RATE = 60.0            # reference rate used when choosing dot_density
        self.dot_speed = 5.0               # deg/s (kept constant across refresh rates)
        self.dot_density = 16.7            # dots/deg²/s at DESIGN_RATE

        # Scale density so dots-per-frame stays similar at the measured refresh rate.
        # This matters on MacBook displays that often run ~120 Hz: without scaling,
        # the same dots/deg²/s would look ~2× sparser per frame than at 60 Hz.
        self.density_adjusted = self.dot_density * (self.measured_rate / self.DESIGN_RATE)

        # Sanity check: these two should be ~equal if the correction worked
        self.dots_per_frame_baseline = self.dot_density / self.DESIGN_RATE
        self.dots_per_frame_adjusted = self.density_adjusted / self.measured_rate

        # Print a quick sanity check
        print("\n=== Density Sanity Check ===")
        print(f"Baseline design: {self.dot_density:.2f} dots/deg²/s @ {self.DESIGN_RATE:.0f} Hz")
        print(f"Measured refresh: {self.measured_rate:.2f} Hz")
        print(f"Adjusted density: {self.density_adjusted:.2f} dots/deg²/s")
        print(f"Dots per frame (baseline): {self.dots_per_frame_baseline:.4f}")
        print(f"Dots per frame (adjusted): {self.dots_per_frame_adjusted:.4f}")
        print(" If these two per-frame values are ~equal, visual density per frame is stable.\n")

    
    # Estimate the true refresh rate from flip timing (after warm-up)
    def measure_refresh_rate(self):

        # Collect timestamps for a fixed number of screen refreshes.
        # win.flip() returns the time (in seconds) when the new frame was actually shown.
        frame_times = []
        for _ in range(240):  # ~2 s at 120 Hz (or ~4 s at 60 Hz)
            frame_times.append(self.win.flip())

        # Convert timestamps -> frame intervals (sec/frame) -> refresh rate (Hz).
        if len(frame_times) > 1:
            intervals = np.diff(frame_times)
            measured_rate = 1.0 / np.mean(intervals)
            print(f"True measured frame rate: {measured_rate:.2f} Hz")
        else:
            measured_rate = 60.0
            print("Could not measure reliably — using default 60 Hz.")
        return measured_rate

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
            dot_density=self.density_adjusted,   
            rng=rdk_rng    
        )

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
            f.write(self._line("Measured refresh rate (Hz):", f"{self.measured_rate:.2f}"))
            f.write(self._line("Design refresh rate used for calculating dot count (Hz):", f"{self.DESIGN_RATE:.2f}"))
            
            # Stimulus parameters 
            f.write(self._section("RDK STIMULUS"))
            f.write(self._line("Dot speed (deg/s):", self.dot_speed))
            f.write(self._line("Baseline dot density (dots/deg²/s):", self.dot_density))
            f.write(self._line("Adjusted dot density (dots/deg²/s):", f"{self.density_adjusted:.2f}"))
            f.write(self._line("Dots per frame (baseline):", f"{self.dots_per_frame_baseline:.4f}"))
            f.write(self._line("Dots per frame (adjusted):", f"{self.dots_per_frame_adjusted:.4f}"))
            f.write(self._line("Total dots:", self.rdk.n_dots))
            f.write(self._line("Field diameter (deg):", self.rdk.field_diameter))

            # Per-sequence dot count 
            dots_per_seq = self.rdk.n_dots // self.rdk.n_sequences
            f.write(self._line("Dots per sequence:", dots_per_seq))
            f.write(self._line("Dots per sequence:", self.rdk.dots_stim.nElements))

            # Environment: versions/backends that can affect timing and input 
            f.write(self._section("ENVIRONMENT"))
            f.write(self._line("PsychoPy version:", psychopy.__version__))
            f.write(self._line("Monitor:", self.monitor_choice))
            f.write(self._line("Window backend:", prefs.general["winType"]))
            f.write(self._line("Keyboard backend:", prefs.hardware["keyboard"]))

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

   

