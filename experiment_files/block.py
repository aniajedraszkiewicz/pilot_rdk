import os
import csv
import numpy as np
from datetime import datetime
from psychopy import visual, core, data 
from .single_trial import Trial
import hashlib
from .helpers import get_block_intro_text



class Block:
    """
    This class runs one block of trials. It uses the Trial class, and also:
        - shows the block intro screen,
        - shows a fixation cross before each trial,
        - controls the between-trial sequence (fixation → choose coherence/direction → run Trial → check correctness → log),
        - marks responses as correct/incorrect,
        - updates QUEST based on correctness (so the next trial’s coherence is chosen adaptively),
        - randomizes direction using a local, seeded RNG (separate from the dot RNG),
        - returns basic QUEST diagnostics at the end of the block.
    
    QUEST (Guénot et al., 2023): Bayesian adaptive method that chooses coherence each trial to estimate the ~82% correct threshold.
    It starts from a prior (initial guess + uncertainty) and updates the threshold estimate after each response.
    Similar to a staircase (harder after correct, easier after incorrect), but QUEST uses all past responses
    via the posterior estimate instead of a fixed up/down rule.
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

        # Initialize QUEST (parameters from Guénot et al., 2023): choose coherence via ML estimate
        quest = data.QuestHandler(
            startVal=0.58,          # prior mean: initial threshold guess (coherence)
            startValSd=0.40,        # prior SD: uncertainty about the initial guess
            pThreshold=0.82,        # target performance level #(82% correct - which is equivalent 
                                    # to a 3 up 1 down standard staircase; PsychoPy QuestHandler documentation)
            gamma=0.5,              # 2AFC guessing rate (left/right → chance = 50%)
            beta=3.5,               # Weibull slope (steepness of the psychometric curve)
            delta=0.01,             # lapse rate (~1% random mistakes)
            nTrials=64,             # number of adaptive trials (set to 64)
            minVal=0.02,            # lower bound for coherence (avoid 0 signal)
            maxVal=0.9,             # upper bound for coherence
            method = 'quantile',    # choose next coherence from the current posterior
            stimScale='linear'
        )


        # Create a Trial object that runs a single trial when called (reused across trials in this block)
        trial_runner = Trial(self.win, self.kb, self.rdk, self.max_stim_sec, debug=self.debug) 
    
        # Collect the running threshold estimate after each QUEST update
        trial_thresholds = []
        
        # Store the first QUEST intensity (coherence) shown in this block
        quest_first_intensity = None


        # ------------------------ Run adaptive trials and log results ------------------------

        # QuestHandler in PsychoPy calls the stimulus level "intensity".
        # With stimScale="linear", QUEST intensities are linear coherence values (minValue-maxValue).
        # So startVal/minVal/maxVal are coherences, and quest.mean() returns a coherence threshold estimate.
        for trial_index, intensity in enumerate(quest):
            
            
            # Convert QUEST "intensity" to the coherence used by the stimulus
            coherence = float(intensity)
            
            # Store only first coherence
            if quest_first_intensity is None:
                quest_first_intensity = coherence

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
            
            # Mark correctness for QUEST (treat timeout as incorrect).
            is_correct = 0 if result["timeout"] else int(result["response_key"] == correct_key)

            # Record the trial outcome in the QUEST algorithm so it can update its internal
            # threshold estimate and choose the coherence for the next trial
            quest.addResponse(is_correct)

            # Store the current threshold estimate after the update (for plots/diagnostics)
            trial_thresholds.append(float(quest.mean()))


            # Build one CSV row: trial settings + response + timing + fixation timing
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "subject_id": self.subject_id,
                "block_no": self.block_no,
                "trial_no": trial_index + 1,
                "condition": "quest_pilot",
                "direction": direction,
                "coherence": float(coherence),
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

        # Compute the final threshold estimate after the last QUEST update.
        final_threshold = float(quest.mean())

        # Compute the range of QUEST intensities used to verify the coherence span
        quest_intensities = [float(x) for x in quest.intensities]
        quest_intensity_min = min(quest_intensities) if quest_intensities else None
        quest_intensity_max = max(quest_intensities) if quest_intensities else None

        # Compute summary diagnostics 
        ci = quest.confInterval()   # Get QUEST confidence interval for the threshold (5th–95th percentile)
        
        diagnostics = {
            "quest_first_intensity": float(quest_first_intensity) if quest_first_intensity is not None else None,
            "quest_intensity_min": float(quest_intensity_min) if quest_intensity_min is not None else None,
            "quest_intensity_max": float(quest_intensity_max) if quest_intensity_max is not None else None,
            "mean": final_threshold,                                  # Final threshold estimate (same as quest.mean() at the end)
            "sd": float(quest.sd()),                                  # Uncertainty (SD) of the current threshold posterior
            "mode": float(quest.mode()),                              # Most likely threshold value (posterior mode)
            "stimuli_used": list(quest.intensities),                  # Coherence values QUEST used across trials
            "responses": list(quest.data),                            # Trial outcomes added to QUEST (1=correct, 0=incorrect/timeout)
            "threshold_estimates": trial_thresholds,                  # Running quest.mean() after each update (trajectory over trials)
            "ci_5_95": (float(ci[0]), float(ci[1])),
            "overall_accuracy": float(np.mean(quest.data)),           # Mean correctness across all QUEST trials
            "last10_accuracy": float(np.mean(list(quest.data)[-10:])) # Mean correctness in the last 10 trials (stability check)
        }

        return diagnostics
    

    # Close the PsychoPy window and exit
    def quit(self):
        self.win.close()
        core.quit()
