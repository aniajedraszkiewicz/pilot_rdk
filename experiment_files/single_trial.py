from psychopy import core
import numpy as np
from datetime import datetime
import csv
import os

class Trial: 
    """
    This class is created for running a single trial:
    - initialize the RDK stimulus with direction and coherence
    - controls stimulus drawing and flipping
    - handles keypress responses
    - measures different sources of time

    Time measurements in 3 layers: 
    
    I. Keypress event timing:
        
        reaction_time - primary RT (s). Time from the first stimulus flip (stimulus onset) to the keypress,
        measured by the ptb or iohub Keyboard clock (k.rt). The keyboard clock is reset on the
        first flip via win.callOnFlip, so this RT is stimulus-locked.

    II. Display flip timing (when the stimulus frame actually appeared/when a frame was refreshed):
    
        global_onset_time – absolute timestamp returned by win.flip() for the first stimulus frame
         
        response_flip_time – absolute timestamp returned by win.flip() for the frame on which the response
        was accepted 

    III. Python loop timing (when the code noticed the response/when the loop ended):
        
        response_detected_time – time (s) on trial_clock when the Python loop noticed/accepted the response.
        trial_clock is also reset on the first flip. This can be slightly later than
        reaction_time because keys are polled once per frame.

        stimulus_on_screen_duration – total time (s) from the first stimulus flip (trial_clock reset) until the loop ends
        (response accepted or timeout)

    Frame-timing measures:
        
        n_long_frames – count of flip-to-flip intervals longer than long_frame_threshold (simple stutter indicator).
        
        max_flip_interval – largest observed flip-to-flip interval within the trial  
        
        response_frame_idx – 0-based index of the displayed frame on which the response was accepted
     
    """

    def __init__(self, win, kb, rdk, max_stim_sec, fixation_stim=None, debug=True):
        self.win = win
        self.kb = kb
        self.rdk = rdk
        self.max_stim_sec = float(max_stim_sec)
        self.fixation_stim = fixation_stim

        # Store debug flag (True = extra diagnostics, False = quiet).
        # Set it in Experiment when creating Block: Block(..., debug=True/False).
        # Block passes it into Trial: Trial(..., debug=self.debug)
        self.debug = bool(debug)

        # Safety: require at least a Keyboard with a resettable clock.
        if not (hasattr(self.kb, "clock") and hasattr(self.kb.clock, "reset")):
            raise TypeError("kb must be psychopy.hardware.keyboard.Keyboard with a resettable clock")
    
    def run_single_trial(self, direction, coherence):
        

        # ------------------------ Set up trial ------------------------

        # Initialize stimulus state for this trial (direction, coherence, internal dot state).
        self.rdk.initialize_rdk_stim(direction=direction, coherence=coherence)
        
        # Create a trial-local clock (will be reset on the first stimulus flip).
        trial_clock = core.Clock()  

        # Clear the key queue now so only responses occurring after the upcoming onset flip can be collected.
        self.kb.clearEvents()       
        
        # Schedule a reset of the keyboard RT clock exactly on the next win.flip()
        self.win.callOnFlip(self.kb.clock.reset)
        
        # Schedule clearing of key events exactly on the same onset flip (drop pre-onset presses)
        self.win.callOnFlip(self.kb.clearEvents)
        
        # Schedule resetting trial_clock exactly on the onset flip (define t=0 at first stimulus frame)
        self.win.callOnFlip(trial_clock.reset)

        # ------------------------ Initialize result variables ------------------------

        # Initialize response-related outputs
        response_key = None
        reaction_time = None
        response_detected_time = None        
        global_onset_time = None
        response_flip_time = None
        response_frame_idx = None
  
        # ------------------------ Initialize counters and debug buffers ------------------------

        # Initialize per-trial frame counters and optional debug collectors.
        frame_count = 0
        frame_stats = [] if self.debug else None           # Store per-frame diagnostic snapshots returned by RDK.get_frame_snapshot() (debug only)
        empirical_coherences = [] if self.debug else None  # Store per-frame empirical coherence values extracted from those snapshots (debug only)
   
        # Read the intended frame rate from the stimulus object (fall back to 60 Hz if missing).
        fps = float(getattr(self.rdk, "frame_rate", 60.0) or 60.0)
        
        # Heuristic threshold: count a frame as "long" if the flip-to-flip interval is > 1.5× the expected frame period (allows some normal jitter above the ideal refresh).
        long_frame_threshold = 1.5 * (1.0 / fps)

        # Initialize flip-to-flip timing trackers to detect dropped/slow frames
        last_flip_time = None       # Store the previous win.flip() timestamp so we can compute flip-to-flip intervals
        n_long_frames = 0           
        max_flip_interval = 0.0     
 
        # ------------------------ Run stimulus loop ------------------------

        # Main per-frame loop: update -> (optional debug) -> draw -> flip -> poll keys -> stop
        while True:

            # Update dot positions/state for the current frame (one sequence update per frame).
            self.rdk.update_rdk_stim()

            # Collect per-frame debug snapshot data 
            if self.debug:
                
                # Get a snapshot for this frame index; replace None with {} to keep append safe
                snap = self.rdk.get_frame_snapshot(frame_count) or {}
                
                frame_stats.append(snap)
                
                # Get empirical coherence for this frame and store it if it is a real number
                emp = snap.get("empirical_coherence")
                if emp is not None and not np.isnan(emp):
                    empirical_coherences.append(emp)

            
            # Draw the dot stimulus into the back buffer (nothing is visible until flip())
            self.rdk.dots_stim.draw()

            # Draw fixation on top of dots every frame (keeps it permanently visible)
            if self.fixation_stim is not None:
                self.fixation_stim.draw()


            # Flip buffers and capture the absolute flip timestamp for this displayed frame
            flip_time = self.win.flip()

            # Compute flip-to-flip timing checks (skip first flip because we have no previous timestamp)
            if last_flip_time is not None:
                flip_interval = flip_time - last_flip_time

                if flip_interval > max_flip_interval:         
                    max_flip_interval = float(flip_interval)

                if flip_interval > long_frame_threshold:
                    n_long_frames += 1
            
            # Save this flip timestamp so the next frame can compute flip_interval
            last_flip_time = flip_time

            # Store the 0-based index of the current displayed frame
            frame_idx = frame_count          
            frame_count += 1

            # Mark onset at the first flip of this trial.
            if global_onset_time is None:
                global_onset_time = flip_time
            
 
            # Check the keyboard for new key events since the last frame (and clear them from the queue).
            keys = self.kb.getKeys(keyList=['left', 'right', 'escape'], clear=True)
            
            if keys:
                # ESC: quit immediately
                if any(k.name == 'escape' for k in keys):
                    self.win.close()
                    core.quit()
                    return None
                
                # Choose the earliest left/right key in this poll (smallest k.rt)
                earliest = None
                for k in keys:
                    if k.name in ('left', 'right'):
                        if earliest is None or k.rt < earliest.rt:
                            earliest = k

                # Accept only if within the response window
                if earliest is not None and earliest.rt <= self.max_stim_sec:
                    response_key = earliest.name                    
                    reaction_time = float(earliest.rt)
                    response_detected_time = trial_clock.getTime()
                    response_flip_time = flip_time
                    response_frame_idx = frame_idx

                    break

            # Timeout: stop if trial_clock (set to 0 at stimulus onset) reaches max_stim_sec
            if trial_clock.getTime() >= self.max_stim_sec:
                break

        # ------------------------ Summarize trial ------------------------

        # Debug prints: mean empirical coherence across frames (if available)
        if self.debug and empirical_coherences:
            mean_empirical = float(np.mean(empirical_coherences))
            print(f"Trial coherence param={coherence}, mean empirical coherence={mean_empirical:.3f}")


        # ------------------------ Write debug CSV ------------------------

        # Write per-frame debug csv (optional)
        if self.debug and frame_stats:
            os.makedirs("results", exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_csv_path = os.path.join("results", f"rdk_frame_debug_{stamp}.csv")
            with open(debug_csv_path, "w", newline="", encoding="utf-8") as f:
                # Write per-frame debug CSV (use all keys seen across frames)
                all_keys = sorted({k for row in frame_stats for k in row.keys()})
                writer = csv.DictWriter(f, fieldnames=all_keys)
                writer.writeheader()
                writer.writerows(frame_stats)
        
        # Total time since stimulus onset until the loop ended (response accepted or timeout).
        stimulus_on_screen_duration = trial_clock.getTime()
        
        # Observed FPS for this trial (sanity check; not the monitor's true refresh rate)
        estimated_fps = (frame_count / stimulus_on_screen_duration) if stimulus_on_screen_duration > 0 else None

        # ------------------------ Create result dict ------------------------

        # Collect all trial-level outputs and timing measures into a single result dict
        result = {
            "direction": int(direction),
            "coherence": float(coherence),
            "response_key": response_key,             # None if timeout
            "reaction_time": reaction_time,           # None if timeout
            "timeout": int(response_key is None),
            "global_onset_time": global_onset_time,
            "response_flip_time": response_flip_time,
            "response_frame_idx": response_frame_idx,
            "response_detected_time": response_detected_time,
            "stimulus_on_screen_duration": stimulus_on_screen_duration,
            "frame_count": frame_count,
            "estimated_fps": estimated_fps,
            "frame_stats": frame_stats,
            "n_long_frames": n_long_frames,
            "max_flip_interval": max_flip_interval,

        }
        return result

    # Close the PsychoPy window and exit 
    def quit(self):
        self.win.close()
        core.quit()