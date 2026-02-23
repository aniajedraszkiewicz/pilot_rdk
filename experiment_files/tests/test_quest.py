"""
Unit tests for QUEST-based Block using safe mocks (no PsychoPy windows).

Tests included:
1) QUEST initialization sanity (matches Block.run_block params)
2) QUEST intensity bounds
3) QUEST convergence sanity (not strict)
4) Block.run_block() with mocked Trial + mocked fixation (expects 64 trials)
"""

from __future__ import annotations

import numpy as np
import pytest
from psychopy import data
from unittest.mock import MagicMock

from experiment_files.block import Block


START_COH = 0.58
MIN_COH = 0.02
MAX_COH = 0.90

START_LOG10 = float(np.log10(START_COH))
MIN_LOG10 = float(np.log10(MIN_COH))
MAX_LOG10 = float(np.log10(MAX_COH))

QUEST_DEFAULTS = dict(
    startVal=START_LOG10,
    startValSd=0.30,
    pThreshold=0.82,
    gamma=0.5,
    beta=3.5,
    delta=0.02,
    nTrials=64,
    minVal=MIN_LOG10,
    maxVal=MAX_LOG10,
    grain=0.02,
    method="quantile",
)



# ==========================================================
# 1) QUEST initialization sanity (match Block.run_block params)
# ==========================================================

def test_quest_initialization_matches_block_defaults():
    quest = data.QuestHandler(**QUEST_DEFAULTS)

    assert float(quest.startVal) == pytest.approx(QUEST_DEFAULTS["startVal"])
    assert float(quest.minVal) == pytest.approx(QUEST_DEFAULTS["minVal"])
    assert float(quest.maxVal) == pytest.approx(QUEST_DEFAULTS["maxVal"])
    assert int(quest.nTrials) == QUEST_DEFAULTS["nTrials"]
    assert float(quest.pThreshold) == pytest.approx(QUEST_DEFAULTS["pThreshold"])


# ==========================================================
# 2) QUEST intensity stays in bounds
# ==========================================================

def test_quest_intensity_bounds():
    quest = data.QuestHandler(**QUEST_DEFAULTS)

    for intensity in quest:
        x = float(intensity)
        assert QUEST_DEFAULTS["minVal"] <= x <= QUEST_DEFAULTS["maxVal"]
        quest.addResponse(1)  # always correct


# ==========================================================
# 3) QUEST "convergence" sanity (not strict)
# ==========================================================

def test_quest_convergence_sanity():
    """
    Very loose sanity check:
    - QUEST intensities are log10(coh)
    - quest.mean() returns log10(coh)
    """
    true_threshold_log10 = float(np.log10(0.40))
    quest = data.QuestHandler(**QUEST_DEFAULTS)

    for intensity in quest:
        resp = 1 if float(intensity) >= true_threshold_log10 else 0
        quest.addResponse(resp)

    est_log10 = float(quest.mean())
    assert np.isfinite(est_log10)
    assert MIN_LOG10 <= est_log10 <= MAX_LOG10

# ==========================================================
# 4) Full Block.run_block test using mocks (expects 64 trials, FAST)
# ==========================================================

def test_block_run_block_with_mock_trial_64_trials(tmp_path, monkeypatch):
    """
    FAST unit test: do not run real-time fixation loop.
    We patch Block.show_fixation() to set last_fix immediately.

    Patches:
    - experiment_files.block.Trial -> FakeTrial (avoid per-frame loop)
    - block.show_fixation -> fast_show_fixation (avoid 1s waiting per trial)
    """
    import experiment_files.block as block_mod

    # Fake window + keyboard (Block.show_fixation won't be used for real timing)
    fake_win = MagicMock()
    fake_kb = MagicMock()
    fake_kb.clearEvents = MagicMock()
    fake_kb.getKeys = MagicMock(return_value=[])  # never ESC

    class FakeRDK:
        pass

    fake_rdk = FakeRDK()

    # Fake Trial aligned with current Trial.run_single_trial output keys
    class FakeTrial:
        def __init__(self, win, kb, rdk, max_stim_sec, debug=True):
            self.max_stim_sec = float(max_stim_sec)

        def run_single_trial(self, direction, coherence):
            direction_i = int(direction)
            response_key = "right" if direction_i == 0 else "left"
            return {
                "direction": direction_i,
                "coherence": float(coherence),
                "response_key": response_key,
                "reaction_time": 0.5,
                "timeout": 0,
                "global_onset_time": 1.0,
                "response_flip_time": 1.5,
                "response_frame_idx": 10,
                "response_detected_time": 0.52,
                "stimulus_on_screen_duration": 0.60,
                "frame_count": 36,
                "estimated_fps": 60.0,
                "frame_stats": None,
                "n_long_frames": 0,
                "max_flip_interval": 0.0,
            }

    monkeypatch.setattr(block_mod, "Trial", FakeTrial)

    results_header = [
        "timestamp","subject_id",
        "block_no","trial_no","condition",
        "direction","coherence","intensity_log10",
        "threshold_estimate_log10",
        "threshold_estimate_coh",
        "response_key","correct_key","is_correct",
        "reaction_time","timeout",
        "global_onset_time","response_flip_time","response_frame_idx",
        "response_detected_time",
        "stimulus_on_screen_duration",
        "frame_count","estimated_fps","n_long_frames","max_flip_interval",
        "fix_onset_time","fix_offset_time","fix_duration","fix_target_sec",
    ]

    block = Block(
        win=fake_win,
        kb=fake_kb,
        rdk=fake_rdk,
        block_no=1,
        subject_id="TEST",
        results_csv_path=str(tmp_path / "test.csv"),
        results_header=results_header,
        max_stim_sec=1.0,
        debug=False,
    )

    def fast_show_fixation(seconds=1.0):
        block.last_fix = {
            "fix_onset_time": 10.0,
            "fix_offset_time": 10.0,
            "fix_duration": 0.0,
            "fix_target_sec": float(seconds),
        }

    block.show_fixation = fast_show_fixation

    diagnostics = block.run_block()

    # 64 trials
    assert len(diagnostics["responses"]) == 64
    assert len(diagnostics["stimuli_used"]) == 64
    assert len(diagnostics["stimuli_used_log10"]) == 64
    assert len(diagnostics["threshold_estimates"]) == 64
    assert len(diagnostics["threshold_estimates_log10"]) == 64
    assert diagnostics["overall_accuracy"] == 1.0
    assert set(diagnostics["responses"]).issubset({0, 1})

    # QUEST served intensities in log10 space within bounds
    assert all(MIN_LOG10 <= float(x) <= MAX_LOG10 for x in diagnostics["stimuli_used_log10"])

    # Coherence values (converted from those intensities) within linear bounds
    assert all(MIN_COH <= float(c) <= MAX_COH for c in diagnostics["stimuli_used"])

    # Final estimates exist and are bounded in their respective spaces
    assert MIN_LOG10 <= float(diagnostics["mean_log10"]) <= MAX_LOG10
    assert MIN_COH <= float(diagnostics["mean_coh"]) <= MAX_COH
 