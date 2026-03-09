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
    

def get_block_intro_text(block_no, lang="en"):
    """Return title and body text for a given block number and language."""
    texts = {
        "en": {
            "titles": {
                1: "Task instructions",
                2: "Experimental_1 (Block 2)",
                3: "Experimental_2 (Block 3)",
                },
            "bodies": {
                1: (
                    "In a moment, you will see dots moving on the screen.\n\n"
                    "The dots will move continuously and their movement may change over time.\n\n"
                    "Your task is to press a key as soon as you think you know the answer: whether most of the dots are moving to the left or to the right.\n\n"
                    "Press the LEFT arrow key if most dots move left.\n"
                    "Press the RIGHT arrow key if most dots move right.\n\n"
                    "After each response, a small cross will appear on the screen. "
                    "Please look at the cross and keep your eyes on it until the next trial starts."
                    ),
                2: "Main task. Trials are randomized.",
                3: "Main task. Trials are randomized.",
            },
            "default_title": "Block {block_no}",
            "default_body": "Press SPACE to begin.",
        },
        "pl": {
            "titles": {
                1: "Instrukcje",
                2: "eksperymentalny_1 (Blok 2)",
                3: "eksperymentalny_2 (Blok 3)",
                },
            "bodies": {
                1: (
                    "Za chwilę zobaczysz na ekranie poruszające się kropki.\n\n"
                    "Kropki będą stale w ruchu, a kierunek ich ruchu może się zmieniać.\n\n"
                    "Twoim zadaniem jest nacisnąć odpowiedni klawisz na klawiaturze tak szybko, jak tylko podejmiesz decyzję: czy większość kropek porusza się w lewo, czy w prawo.\n\n"
                    "Naciśnij lewą strzałkę, jeśli uważasz, że większość kropek porusza się w lewo. \n"
                    "Naciśnij prawą strzałkę, jeśli uważasz, że większość kropek porusza się w prawo.\n\n"
                    "Po każdej odpowiedzi na ekranie pojawi się mały biały krzyżyk. "
                    "Patrz na krzyżyk do momentu ponownego pojawienia się kropek."
                    ),
                2: "Główne zadanie.",
                3: "Główne zadanie.",
            },
            "default_title": "Blok {block_no}",
            "default_body": "Naciśnij spację, aby rozpocząć.",
        }  
    }

    if lang not in texts:
        raise ValueError(f"Unsupported language: {lang}")

    title = texts[lang]["titles"].get(
        block_no,
        texts[lang]["default_title"].format(block_no=block_no)
    )
    body = texts[lang]["bodies"].get(
        block_no,
        texts[lang]["default_body"]
    )

    return title, body

 

def get_block_outro_text(block_no, lang="en"):
    """Return title and body text shown after a block is finished."""
    texts = {
        "en": {
            "titles": {
                1: "Task completed",
                2: "End of block",
                3: "End of block",
            },
            "bodies": {
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
            },
            "default_title": "End of block",
            "default_body": "Press SPACE to continue.",
        },

        "pl": {
            "titles": {
                1: "Koniec zadania",
                2: "Koniec bloku",
                3: "Koniec bloku",
            },
            "bodies": {
                1: (
                    "Dziękuję za udział w badaniu.\n\n"
                    "Proszę poinformować osobę prowadzącą badanie, że zadanie zostało ukończone."
                ),
                2: (
                    "Ten blok dobiegł końca.\n\n"
                    "Możesz zrobić krótką przerwę, jeśli tego potrzebujesz.\n\n"
                    "Naciśnij spację, kiedy będziesz gotowy/gotowa kontynuować badanie."
                ),
                3: (
                    "Ten blok dobiegł końca.\n\n"
                    "Możesz zrobić krótką przerwę, jeśli tego potrzebujesz.\n\n"
                    "Naciśnij spację, kiedy będziesz gotowy/gotowa kontynuować badanie."
                ),
            },
            "default_title": "Koniec bloku",
            "default_body": "Naciśnij spację, żeby kontynuować.",
        }
    }



    if lang not in texts:
        raise ValueError(f"Unsupported language: {lang}")

    title = texts[lang]["titles"].get(block_no, texts[lang]["default_title"])
    body = texts[lang]["bodies"].get(block_no, texts[lang]["default_body"])

    return title, body