# RDK Experiment – pilot QUEST study (PsychoPy)

This repository contains a PsychoPy-based pilot experiment investigating perceptual decision-making with a Random Dot Kinematogram (RDK).

This **pilot version uses QUEST** (a Bayesian adaptive procedure) to select motion coherence values across trials and estimate an individual threshold. The goal is to validate timing, logging, and stimulus behavior, and to check whether the adaptive procedure behaves sensibly before switching to a fixed-trial main study.

---

## Folder overview

### `experiment_files/`
Main folder containing the experiment code, input files, outputs, and tests.

Core logic:
- **`experiment.py`** – high-level controller: environment setup, monitor/window creation, participant info collection, refresh-rate measurement, parameter definition (including density correction), stimulus initialization, and run summary saving.
- **`block.py`** – runs a single block: shows instructions, runs fixation and trial flow, handles QUEST updates, and appends results to CSV.
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
- **`test_quest.py`** – unit tests for the pilot’s QUEST procedure (no real windows): parameter sanity, intensity bounds, basic convergence, and a fast mocked `Block.run_block()` (64 trials), including automatic handling of QUEST output scale (linear vs log10).

---

## Running the experiment

From the repository root:

```bash
python -m experiment_files.experiment


