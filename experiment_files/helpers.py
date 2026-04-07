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
                    "Your task:\n"
                    "Decide whether MOST of the dots are moving LEFT or RIGHT.\n"
                    "Press the corresponding key as soon as you have made your decision.\n\n"
                    "Press the LEFT arrow key if most dots are moving left.\n"
                    "Press the RIGHT arrow key if most dots are moving right.\n\n"
                    "The task has four parts with short breaks in between.\n"
                    "At the beginning, after each response you will receive feedback "
                    "telling you whether your answer was correct or incorrect.\n"
                    "In the later parts, no feedback will be given.\n\n"
                    "Sometimes the task will be easier, sometimes harder — this is normal.\n\n"
                    "Important: keep your eyes on the green dot in the center of the screen at all times.\n"
                    "Do not track individual moving dots."
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
                    "Twoje zadanie:\n"
                    "zdecyduj, czy WIĘKSZOŚĆ kropek porusza się w LEWO, czy w PRAWO. \n"
                    "Naciśnij odpowiedni klawisz tak szybko, jak tylko podejmiesz decyzję. \n\n"
                    "Naciśnij lewą strzałkę, jeśli większość kropek porusza się w lewo.\n"
                    "Naciśnij prawą strzałkę, jeśli większość kropek porusza się w prawo.\n\n"
                    "Zadanie składa się z 4 części z krótkimi przerwami między nimi.\n"
                    "Na początku po każdej odpowiedzi otrzymasz informację zwrotną, "
                    "czy Twoja odpowiedź była poprawna, czy niepoprawna.\n"
                    "W kolejnych częściach ta informacja nie będzie wyświetlana.\n\n"
                    "Czasem zadanie będzie łatwiejsze, czasem trudniejsze — to naturalne.\n\n"
                    "Ważne: przez cały czas patrz na zieloną kropkę pośrodku ekranu.\n"
                    "Nie śledź wzrokiem pojedynczych poruszających się kropek."
                    
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