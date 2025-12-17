"""
Unit tests for QUEST-based Block using safe mocks (no PsychoPy windows).

Tests included:
1) QUEST initialization sanity (matches Block.run_block params)
2) QUEST intensity bounds
3) QUEST convergence sanity (not strict)
4) Block.run_block() with mocked Trial + mocked fixation (expects 64 trials, FAST)
   + auto-detect QUEST mean() scale (linear vs log10) and apply correct bounds check
"""

from __future__ import annotations

import numpy as np
import pytest
from psychopy import data
from unittest.mock import MagicMock

from experiment_files.block import Block


QUEST_DEFAULTS = dict(
    startVal=0.58,
    startValSd=0.40,
    pThreshold=0.82,
    gamma=0.5,
    beta=3.5,
    delta=0.01,
    nTrials=64,
    minVal=0.02,
    maxVal=0.9,
    method="quantile",
)


# --------------------------
# Scale helpers
# --------------------------

def infer_quest_output_scale(mean_val: float, min_val: float, max_val: float) -> str:
    """
    Infer whether QuestHandler.mean() is on:
      - "linear": mean is already a coherence-like value in [minVal, maxVal]
      - "log10": mean is log10(coherence), so 10**mean is in [minVal, maxVal]
      - "ambiguous": can't decide robustly
    """
    # Negative values cannot be linear coherence; try log10 interpretation first.
    if mean_val < 0:
        coh = 10 ** mean_val
        return "log10" if (min_val <= coh <= max_val) else "ambiguous"

    in_linear = (min_val <= mean_val <= max_val)
    coh_from_log10 = 10 ** mean_val
    in_log10 = (min_val <= coh_from_log10 <= max_val)

    if in_linear and not in_log10:
        return "linear"
    if in_log10 and not in_linear:
        return "log10"
    return "ambiguous"


def assert_mean_within_bounds_under_inferred_scale(mean_val: float, min_val: float, max_val: float) -> str:
    """
    Assert that QUEST mean is within bounds under the inferred scale.
    Returns the inferred scale ("linear" or "log10") for optional debugging.
    """
    scale = infer_quest_output_scale(mean_val, min_val, max_val)

    if scale == "linear":
        assert min_val <= mean_val <= max_val, (
            f"QUEST mean() expected linear within [{min_val}, {max_val}], got {mean_val}"
        )
        return scale

    if scale == "log10":
        coh = 10 ** mean_val
        assert min_val <= coh <= max_val, (
            f"QUEST mean() appears log10; 10**mean={coh} not within [{min_val}, {max_val}] "
            f"(mean={mean_val})"
        )
        return scale

    coh = 10 ** mean_val
    raise AssertionError(
        "Could not infer QUEST mean() scale robustly. "
        f"mean={mean_val}, 10**mean={coh}, bounds=[{min_val},{max_val}]"
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
    true_threshold = 0.40
    quest = data.QuestHandler(**QUEST_DEFAULTS)

    for intensity in quest:
        resp = 1 if float(intensity) >= true_threshold else 0
        quest.addResponse(resp)

    est = float(quest.mean())
    assert np.isfinite(est)
    # Don't assume scale here; just confirm it's interpretable
    assert infer_quest_output_scale(est, QUEST_DEFAULTS["minVal"], QUEST_DEFAULTS["maxVal"]) in ("linear", "log10")


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
        "direction","coherence",
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
    assert len(diagnostics["threshold_estimates"]) == 64
    assert diagnostics["overall_accuracy"] == 1.0
    assert set(diagnostics["responses"]).issubset({0, 1})

    # Intensities (stimuli) always within bounds (these are what QUEST actually served)
    assert all(
        QUEST_DEFAULTS["minVal"] <= float(x) <= QUEST_DEFAULTS["maxVal"]
        for x in diagnostics["stimuli_used"]
    )

    # Mean: detect scale and validate bounds under that scale
    mean_val = float(diagnostics["mean"])
    assert np.isfinite(mean_val)
    scale = assert_mean_within_bounds_under_inferred_scale(
        mean_val, QUEST_DEFAULTS["minVal"], QUEST_DEFAULTS["maxVal"]
    )

    # Only visible if you run pytest with -s
    print(f"\n[QUEST] mean={mean_val:.6f} inferred_scale={scale} (10**mean={10**mean_val:.6f})")
