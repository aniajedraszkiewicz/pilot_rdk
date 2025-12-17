# === HELPER FUNCTIONS ===
import numpy as np
import csv
from psychopy import visual, core


def load_trials(csv_path):
    """Read trials.csv and return a list of dicts (typed values)."""
    trials = []
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            trials.append({
                "block_no": int(row["block_no"]),
                "trial_no": int(row["trial_no"]),
                "condition": row["condition"],
                "direction": int(row["direction"]),
                "coherence": float(row["coherence"]),
                "display_param": row.get("display_param", ""),
            })
    return trials
    

def get_block_intro_text(block_no):
    """Return title and body text for a given block number (no PsychoPy)."""
    title_by_block = {
        1: "Practice (Block 1)",
        2: "Experimental_1 (Block 2)",
        3: "Experimental_2 (Block 3)",
    }
    body_by_block = {
        1: "You’ll do a few easy trials to get familiar with the task.",
        2: "Main task. Trials are randomized.",
        3: "Main task. Trials are randomized.",
    }

    title = title_by_block.get(block_no, f"Block {block_no}")
    body  = body_by_block.get(block_no, "Press SPACE to begin.")

    return title, body
