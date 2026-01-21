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
    """Return title and body text for a given block number."""
    title_by_block = {
        1: "Task instructions",
        2: "Experimental_1 (Block 2)",
        3: "Experimental_2 (Block 3)",
    }
    body_by_block = {
        1: "In a moment, you will see dots moving on the screen.\n\n"
    "The dots will move continuously and their movement may change over time.\n\n"
    "Your task is to decide, as quickly and accurately as possible, whether most of the dots are moving to the left or to the right.\n\n"
    "Press the LEFT arrow key if most dots move left.\n"
    "Press the RIGHT arrow key if most dots move right.",
        2: "Main task. Trials are randomized.",
        3: "Main task. Trials are randomized.",
    }

    title = title_by_block.get(block_no, f"Block {block_no}")
    body  = body_by_block.get(block_no, "Press SPACE to begin.")

    return title, body

def get_block_outro_text(block_no):
    """Return title and body text shown after a block is finished."""
    
    title_by_block = {
        1: "Task completed",
        2: "End of block",
        3: "End of block",
    }

    body_by_block = {
        1: (
            "Thank you for your participation.\n\n"
            "Please inform the person conducting the study."
        ),
        2: (
            "You have completed this block.\n\n"
            "You can take a short break if needed.\n\n"
            "Press SPACE when you are ready to continue."
        ),
        3: (
            "You have completed this block.\n\n"
            "You can take a short break if needed.\n\n"
            "Press SPACE when you are ready to continue."
        ),
    }

    title = title_by_block.get(block_no, "End of block")
    body  = body_by_block.get(
        block_no,
        "Press SPACE to continue."
    )

    return title, body