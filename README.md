# RDK Experiment – pilot QUEST study (PsychoPy)

This repository contains a PsychoPy-based pilot experiment investigating perceptual decision-making with a Random Dot Kinematogram (RDK).

This **pilot version uses QUEST** (a Bayesian adaptive procedure) to select motion coherence values across trials and estimate an individual threshold. The goal is to validate timing, logging, and stimulus behavior, and to check whether the adaptive procedure behaves sensibly before switching to a fixed-trial main study.

---

## Folder overview

### `experiment_files/`
Main folder containing the experiment code, input files, outputs, and tests.

Core logic:
- **`experiment.py`** – high-level controller: environment setup, monitor/window creation, participant info collection, parameter definition (including density correction), stimulus initialization, and run summary saving.
- **`block.py`** – runs all trials in the block: shows instructions, runs fixation and trial flow, handles QUEST updates, and appends results to CSV.
- **`single_trial.py`** – runs a single trial: stimulus loop (draw/flip), response collection, and timing measurements (stimulus-locked RTs and flip timing diagnostics).
- **`rdk_stim.py`** – RDK stimulus generation and frame-by-frame dot updates.
- **`helpers.py`** – helper utilities (e.g., instruction text, loading trial files if used).

### `data/`
- **`trials_exp.csv`** – trial definitions and block structure (used if running fixed trials; the current QUEST pilot may not rely on this file).

### `results/` (generated)
- Participant outputs (CSV logs, run summaries, diagnostic plots).  
  **Not tracked in Git.**

### `tests/`
Automated tests verifying correctness, stability, and reproducibility.

- **`test_block_append.py`** – checks whether trial rows are appended to the CSV correctly.
- **`test_exp_core.py`** – headless-safe tests for Experiment setup (window creation via stubs, refresh-rate math, summary writing, etc.).
- **`test_rdk_stim.py`** – validates RDK initialization, dot updating, and sequence cycling.
- **`test_single_trial.py`** – verifies timing, keyboard handling, and single-trial logic.
- **`test_quest.py`** – unit tests for the pilot’s QUEST procedure (no real windows): parameter sanity, intensity bounds, basic convergence, and a fast mocked `Block.run_block()` (64 trials)

---

## Running the experiment

The repository provides two launcher scripts that run the same experiment implementation with different keyboard backends.

The code was developed using the **VS Code editor** with a manually created **Python virtual environment**. On macOS, only the **ioHub** keyboard backend worked reliably in this setup. The **PTB (Psychtoolbox)** backend, which is the default used by PsychoPy, requires additional dependencies from the MATLAB Psychtoolbox that were not accessible in this environment during development (January 2026).

Therefore, the experiment can be launched using two different approaches:

### Running from VS Code / Python environment (ioHub backend)

From the project root:

```bash
python run_experiment_iohub.py
```

### Running from PsychoPy Standalone (PTB backend)

If **PsychoPy Standalone** is installed, open `run_experiment_ptb.py` in **PsychoPy Coder** and run the script. 
In this case, the PTB keyboard backend will be used.

---
## Instruction language

Participant instructions are defined in:

`experiment_files/helpers.py`

Two language versions are currently implemented:

- `en` – English  
- `pl` – Polish  

The language used during the experiment is specified when initializing the `Block` class (in `block.py`):

```python
Block(..., lang="pl")
```

---
## Running unit tests

Automated tests are implemented using **pytest**.  
They verify the correctness and reproducibility of the experiment logic without opening real PsychoPy windows.

The tests use **mocked components** (fake windows, keyboards, and stimuli) so they can run safely and quickly in a headless environment.

### Activate the virtual environment

Before running tests, activate the Python virtual environment used for development:

```bash
source venv/bin/activate
```

### Run all tests 

From the **project root** directory:

```bash
python -m pytest -v
```
Running tests from the project root ensures that the experiment_files package can be imported correctly.

### Run tests for a specific module

Example:

```bash
python -m pytest experiment_files/tests/test_quest.py -v
```

