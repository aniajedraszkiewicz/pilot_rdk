import os
import csv
import numpy as np
from datetime import datetime
from psychopy import visual, core, data 
from .single_trial import Trial
import hashlib
from .helpers import get_block_intro_text, get_block_outro_text


  
def compute_bias_metrics_right_positive(true_labels, pred_labels):
    """
    Compute post-hoc response bias diagnostics treating Right as the positive class. 
    Intended ONLY for quick sanity checks ("is the participant over-responding Right?"); this is not a standard classification task; the primary goal of the experiment is to
    estimate the relationship between stimulus properties and the probability of a correct response.
    true_labels: 1 if direction==0 (Right) else 0 (Left-180°)(reflects what was shown on the screen)
    pred_labels: 1 if response_key=='right' else 0 (reflects what the participant reported)  
    
    Computed quantities
    -------------------
    tp (true positives):
        Right stimulus correctly identified as Right

    fp (false positives):
        Left incorrectly reported as Right

    fn (false negatives):
        Right incorrectly reported as Left

    precision (Right):
        P(Right stimulus | Right response)
        → When the participant says "Right", how often is that correct?

    recall (Right):
        P(Right response | Right stimulus)
        → When the stimulus is Right, how often does the participant say "Right"?

    f1 (Right):
        Harmonic mean of precision and recall.
        → Single summary of Right-response bias consistency

    p_right_stimulus:
        Base rate of Right stimuli in the trials

    p_right_response:
        Base rate of Right responses given by the participant
    """
    y_true = np.asarray(true_labels, dtype=int)
    y_pred = np.asarray(pred_labels, dtype=int)

    if y_true.size == 0:
        return {
            "bias_precision_right": np.nan,
            "bias_recall_right": np.nan,
            "bias_f1_right": np.nan,
            "p_right_stimulus": np.nan,
            "p_right_response": np.nan,
        }

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else np.nan

    return {
        "bias_precision_right": float(precision),
        "bias_recall_right": float(recall),
        "bias_f1_right": float(f1),
        "p_right_stimulus": float(np.mean(y_true == 1)),
        "p_right_response": float(np.mean(y_pred == 1)),
    }

class Block:
    """
    This class runs one block of trials. It uses the Trial class, and also:
        - shows the block intro screen,
        - shows a fixation cross before each trial,
        - controls the between-trial sequence (fixation → choose coherence/direction → run Trial → check correctness → log),
        - marks responses as correct/incorrect,
        - updates the QUEST posterior based on trial outcomes to adaptively select
          the coherence level for the next trial,
        - randomizes direction using a local, seeded RNG (separate from the dot RNG),
        - returns basic QUEST diagnostics at the end of the block.
    
    QUEST (Guénot et al., 2023): Bayesian adaptive method that chooses coherence each trial to estimate
    the coherence threshold corresponding to a target performance level of approximately 82% correct.
    It starts from a prior (initial guess + uncertainty) and updates the posterior threshold estimate after each response.
    Similar to a staircase (harder after correct, easier after incorrect), but QUEST uses all past responses
    via the posterior mean estimate instead of a fixed up/down rule.
    """

    # ------------------------ Store block configuration ------------------------
    
    def __init__(self, win, kb, rdk, block_no, subject_id, results_csv_path, results_header, max_stim_sec, debug=True):
        
        # Store core PsychoPy objects shared by this block
        self.win = win
        self.kb = kb
        self.rdk = rdk

        # Store debug flag
        self.debug = bool(debug)
        
        # Store block identifiers
        self.block_no = block_no
        self.subject_id = subject_id

        # Store CSV logging settings
        self.results_csv_path = results_csv_path
        self.results_header = results_header
        
        # Store per-trial time limit (seconds)
        self.max_stim_sec = max_stim_sec
       
    # ------------------------ Seed block-level RNG ------------------------

        # Seed a local RNG for block-level design choices (e.g., direction), separate from the dot RNG.
        seed_str = f"{subject_id}_BLOCK_{block_no}".encode("utf-8") # build seed text
        digest = hashlib.sha256(seed_str).hexdigest()               # hash -> stable hex
        seed = int(digest[:8], 16)                                  # take 32-bit int seed
        self.block_seed = seed                                      # store for logging
        self.rng = np.random.default_rng(seed)                      # make block RNG

    
    # ------------------------ Show intro screen ------------------------

    # Display the block intro screen (press SPACE to start, ESC to quit) 
    def show_intro(self):
             # Get the title/body text for this block number
            title, body = get_block_intro_text(self.block_no)

            # Build a text stimulus to display instructions
            msg = visual.TextStim(
                self.win,
                text=f"{title}\n\n{body}\n\nPress SPACE to start\n(ESC to quit)",
                height=0.9,
                color='white',
                wrapWidth=20
            )

            # Clear buffered keys before checking for new key presses
            self.kb.clearEvents()

            while True:
                # Quit immediately if ESC is pressed
                if self.kb.getKeys(keyList=['escape'], clear=True):
                    self.win.close()
                    core.quit()
                    return
                
                # Draw the message and present it on screen
                msg.draw()
                self.win.flip()

                # Start the block when SPACE or RETURN is pressed
                if self.kb.getKeys(keyList=['space', 'return'], clear=True):
                    break

    # ------------------------ Show outro screen ------------------------

    def show_outro(self):
        """Display the block outro screen"""
        
        title, body = get_block_outro_text(self.block_no)

        msg = visual.TextStim(
            self.win,
            text=f"{title}\n\n{body}\n\nPress SPACE to continue",
            height=0.9,
            color="white",
            wrapWidth=20,
        )

        self.kb.clearEvents()

        while True:

            msg.draw()
            self.win.flip()

            if self.kb.getKeys(keyList=["space", "return"], clear=True):
                break

    # ------------------------ Show fixation and store timing ------------------------

    # Show a fixation cross for a fixed duration and store its timing
    def show_fixation(self, seconds=1.0):
        txt = visual.TextStim(self.win, text='+', height=1.2, color='white')
        clock = core.Clock()
        clock.reset()
        
        # Clear keys so earlier presses don’t carry into fixation
        self.kb.clearEvents()

        # Store first/last flip times to check how long fixation really stayed on screen.
        # The requested duration is a target; the true duration is quantized by frames
        fix_onset_time = None
        fix_offset_time = None

        # Keep showing fixation until this timer reaches `seconds` (time since clock.reset()).
        while clock.getTime() < float(seconds):
            # Allow quiting during fixation
            if self.kb.getKeys(keyList=["escape"], clear=True):
                self.win.close()
                core.quit()
                return

            # Draw the fixation cross and show it on the next screen refresh (flip)
            txt.draw()
            fix_offset_time = self.win.flip() # store the timestamp of this flip

            # Save the timestamp of the first refresh that showed fixation (set once, then keep it)
            if fix_onset_time is None:
                fix_onset_time = fix_offset_time

        # Save fixation timing so you can log/check the actual on-screen duration later
        self.last_fix = {
            "fix_onset_time": fix_onset_time,   # first fixation flip time
            "fix_offset_time": fix_offset_time, # last fixation flip time
            "fix_duration": (fix_offset_time - fix_onset_time) if fix_onset_time is not None else None,
            "fix_target_sec": float(seconds), # requested duration (target)
        }
    
    # ------------------------ Append one trial row to CSV ------------------------

    # Append one results row to a CSV file.
    # If the file does not exist yet, create it and write the header first
    def append_log_row(self, csv_out_path, row, header):
        new_file = not os.path.exists(csv_out_path)
        with open(csv_out_path, "a", newline='', encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if new_file:
                w.writeheader()
            w.writerow(row)
    
    # ------------------------ Initialize QUEST ------------------------

    # Run one QUEST block: loop over adaptive trials, update QUEST from correctness, and log each trial
    def run_block(self):  

        # Coherence shown to participants (linear units)
        start_coh = 0.58
        min_coh = 0.02
        max_coh = 0.9

        # PsychoPy QuestHandler "intensity" is log10 units, so convert coherence -> log10(coherence)
        start_intensity_log10 = float(np.log10(start_coh))
        min_intensity_log10 = float(np.log10(min_coh))
        max_intensity_log10 = float(np.log10(max_coh))

        # startValSd must be in same units as startVal (log10 units here)
        start_intensity_sd_log10 = 0.30

        # Initialize QUEST (parameters from Guénot et al., 2023): choose coherence via quantile selection
        quest = data.QuestHandler(
            startVal=start_intensity_log10,             # prior mean: initial threshold guess (log10 coherence)
            startValSd=start_intensity_sd_log10,        # prior SD in log10: uncertainty about the initial guess
            pThreshold=0.82,                            # target performance level (82% correct - which is equivalent 
                                                        # to a 3 up 1 down standard staircase; PsychoPy QuestHandler documentation)
            gamma=0.5,                                  # 2AFC guessing rate (left/right → chance = 50%)
            beta=3.5,                                   # Weibull slope (steepness of the psychometric curve)
            delta=0.02,                                 # lapse rate (~2% random mistakes)
            nTrials=64,                                 # number of adaptive trials (set to 64)
            minVal=min_intensity_log10,                 # lower bound for log10 coherence (avoid 0 signal)
            maxVal=max_intensity_log10,                 # upper bound for log10 coherence
            grain=0.02,                                # step size in intensity units (log10)
            method = 'quantile',                        # choose next intensity from the current posterior
        )


        # Create a Trial object that runs a single trial when called (reused across trials in this block)
        trial_runner = Trial(self.win, self.kb, self.rdk, self.max_stim_sec, debug=self.debug) 
    
        # Collect the running threshold estimate after each QUEST update
        threshold_estimates_history_log10 = []
        threshold_estimates_history_coh = []

        # Collect labels for post-hoc bias diagnostics (exclude timeouts)
        bias_y_true = []  # 1 = Right stimulus (direction==0), 0 = Left stimulus (direction==180)
        bias_y_pred = []  # 1 = Right response, 0 = Left/other response

        # ------------------------ Run adaptive trials and log results ------------------------

        # QuestHandler calls the stimulus level "intensity".
        # Here, intensity = log10(coherence), so convert via coherence = 10**intensity.
        # startVal / minVal / maxVal are specified in log10(coherence) units.
        # quest.mean() returns a threshold estimate in log10(coherence).
        for trial_index, intensity in enumerate(quest):

            # Convert QUEST "intensity" to the coherence used by the stimulus
            intensity = float(intensity)
            coherence = float(10 ** intensity)

            # Reset fixation timing from the previous trial, then show fixation and store its timing in self.last_fix
            self.last_fix = None
            self.show_fixation(seconds=1.0)

            # Store fixation timing for logging; stop if fixation was interrupted (e.g., ESC)
            fix = self.last_fix
            if fix is None:
                return 
            
            # Clear any keypresses during fixation 
            self.kb.clearEvents()  

            # Randomize direction
            direction = 0 if self.rng.random() < 0.5 else 180

            # Run RDK trial with selected coherence
            result = trial_runner.run_single_trial(direction, coherence)
            
            # Stop the block if the trial returned None (ESC inside Trial)
            if result is None:
                return  

            # Define the correct response key for this direction (0° = right, 180° = left).
            correct_key = 'right' if direction == 0 else 'left'
            
            # Collect data for post-hoc bias diagnostics (exclude timeouts)
            if not result.get("timeout"):
                true_label = 1 if direction == 0 else 0
                resp = (result.get("response_key") or "").lower()
                pred_label = 1 if resp == "right" else 0
                bias_y_true.append(true_label)
                bias_y_pred.append(pred_label)
            
            # Mark correctness for QUEST (treat timeout as incorrect).
            is_correct = 0 if result["timeout"] else int(result["response_key"] == correct_key)

            # Record trial correctness in QUEST so the posterior threshold estimate
            # is updated and the next coherence level can be selected adaptively
            quest.addResponse(is_correct, intensity=intensity)

            # Store current threshold estimate AFTER the update (posterior mean; log10 units)
            threshold_estimate_log10 = float(quest.mean())
            threshold_estimate_coh = float(10 ** threshold_estimate_log10)

            # Store the trajectory of threshold estimates
            threshold_estimates_history_log10.append(threshold_estimate_log10)
            threshold_estimates_history_coh.append(threshold_estimate_coh)

            # Build one CSV row: trial settings + response + timing + fixation timing
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "subject_id": self.subject_id,
                "block_no": self.block_no,
                "trial_no": trial_index + 1,
                "condition": "quest_pilot",
                "direction": direction,
                "coherence": float(coherence), # motion coherence shown on this trial (linear units used by the stimulus)
                "intensity_log10": float(intensity), # QUEST intensity for this trial = log10(coherence)
                "threshold_estimate_log10": threshold_estimate_log10,  # current QUEST threshold estimate after this trial (intensity = log10 coherence)
                "threshold_estimate_coh": threshold_estimate_coh, # same threshold estimate converted to linear coherence
                "response_key": result.get("response_key"),
                "correct_key": correct_key,
                "is_correct": is_correct,
                "reaction_time": result.get("reaction_time"),
                "timeout": result.get("timeout"),
                "global_onset_time": result.get("global_onset_time"),
                "response_flip_time": result.get("response_flip_time"),
                "response_frame_idx": result.get("response_frame_idx"),
                "response_detected_time": result.get("response_detected_time"),
                "stimulus_on_screen_duration": result.get("stimulus_on_screen_duration"),
                "frame_count": result.get("frame_count"),
                "estimated_fps": result.get("estimated_fps"),
                "n_long_frames": result.get("n_long_frames"),
                "max_flip_interval": result.get("max_flip_interval"),
                "fix_onset_time": fix["fix_onset_time"],
                "fix_offset_time": fix["fix_offset_time"],
                "fix_duration": fix["fix_duration"],
                "fix_target_sec": fix["fix_target_sec"],
            }
            
            # Append this trial row to the results CSV 
            self.append_log_row(self.results_csv_path, row, self.results_header)
            
       
        # ------------------------ Summarize threshold and return diagnostics ------------------------

        # Final threshold estimates after the last QUEST update. 
        # QUEST outputs are in log10(coherence); converted to coherence for reporting.
        # Note: not all quantities computed here are necessarily written to the text summary;
        # this section can be extended to compute additional diagnostics as needed.
        final_mean_log10 = float(quest.mean())      # posterior mean threshold (log10)
        final_mode_log10 = float(quest.mode())      # posterior mode (log10)
        final_sd_log10 = float(quest.sd())          # posterior SD (log10)

        # Convert threshold estimates from log10 units to linear coherence
        final_mean_coh = float(10 ** final_mean_log10)
        final_mode_coh = float(10 ** final_mode_log10)

        # 5–95% credible interval from QUEST (returned in log10 units)
        ci_log10 = quest.confInterval()
        ci_coh = (float(10 ** float(ci_log10[0])), float(10 ** float(ci_log10[1])))

        # Intensities actually presented during the block
        intensities_log10 = [float(x) for x in quest.intensities] #log10 unit
        stimuli_coh = [float(10 ** x) for x in intensities_log10] #linear coherence unit

        
        # Compute summary diagnostics 
        bias_metrics = compute_bias_metrics_right_positive(bias_y_true, bias_y_pred)

        diagnostics = {
            "quest_start_coh": float(start_coh),
            "quest_min_coh": float(min_coh),
            "quest_max_coh": float(max_coh),
            "quest_start_intensity_log10": float(quest.startVal),
            "quest_start_intensity_sd_log10": float(quest.startValSd),
            "quest_min_intensity_log10": float(quest.minVal),
            "quest_max_intensity_log10": float(quest.maxVal),
            "quest_pThreshold": float(quest.pThreshold),
            "quest_gamma": float(quest.gamma),
            "quest_beta": float(quest.beta),
            "quest_delta": float(quest.delta),
            "quest_nTrials": int(quest.nTrials),
            "quest_method": quest.method,
            "quest_grain": quest.grain,   
            "coh_min_used": float(min(stimuli_coh)) if stimuli_coh else None,
            "coh_max_used": float(max(stimuli_coh)) if stimuli_coh else None,
            "mean_coh": final_mean_coh,
            "mode_coh": final_mode_coh,
            "mean_log10": final_mean_log10,
            "mode_log10": final_mode_log10, 
            "sd_log10": final_sd_log10,
            "stimuli_used": stimuli_coh,
            "stimuli_used_log10": intensities_log10,
            "responses": list(quest.data),
            "threshold_estimates": threshold_estimates_history_coh,
            "threshold_estimates_log10": threshold_estimates_history_log10,
            "ci_5_95_coh": ci_coh,
            "ci_5_95_log10": ci_log10,
            "overall_accuracy": float(np.mean(quest.data)),
            "last10_accuracy": float(np.mean(list(quest.data)[-10:])),
            "bias_precision_right": bias_metrics["bias_precision_right"],
            "bias_recall_right": bias_metrics["bias_recall_right"],
            "bias_f1_right": bias_metrics["bias_f1_right"],
            "p_right_stimulus": bias_metrics["p_right_stimulus"],
            "p_right_response": bias_metrics["p_right_response"],
        }

        return diagnostics
    
    
    # Close the PsychoPy window and exit
    def quit(self):
        self.win.close()
        core.quit()
