# tests/test_single_trial.py
import pytest

from experiment_files.single_trial import Trial


# -----------------------------
# Test doubles (headless-safe)
# -----------------------------

class DummyWin:
    """
    Fake PsychoPy window that:
    - stores callOnFlip callbacks and runs them exactly on flip()
    - advances a deterministic "flip time" by dt each flip
    - returns that flip time as win.flip() timestamp
    """
    def __init__(self, dt=1/60):
        self.dt = float(dt)
        self.t = 0.0
        self._callbacks = []
        self.flips = 0

    def callOnFlip(self, func):
        self._callbacks.append(func)

    def flip(self):
        # run scheduled callbacks exactly on flip (PsychoPy semantics)
        callbacks, self._callbacks = self._callbacks, []
        for fn in callbacks:
            fn()

        self.t += self.dt
        self.flips += 1
        return self.t

    def close(self):
        pass


class DummyClock:
    """Simple clock with reset() and getTime() used by Trial."""
    def __init__(self):
        self._t = 0.0

    def reset(self):
        self._t = 0.0

    def getTime(self):
        return self._t

    # internal: let test advance time
    def _advance(self, dt):
        self._t += float(dt)


class DummyKb:
    """
    Fake keyboard that returns a predefined per-frame list of key events.
    Also provides kb.clock.reset for stimulus-locked RT.
    """
    def __init__(self, per_poll_keys=None):
        # per_poll_keys: list where each element is the list returned by getKeys on that poll
        self._per_poll = list(per_poll_keys or [])
        self.clock = DummyClock()

    def clearEvents(self):
        pass

    def getKeys(self, keyList=None, clear=True):
        if self._per_poll:
            return self._per_poll.pop(0)
        return []


class KeyEvt:
    """Minimal key event object with .name and .rt as expected by Trial."""
    def __init__(self, name, rt):
        self.name = name
        self.rt = rt


class DummyRDK:
    """
    Minimal RDK stub for Trial:
    - initialize_rdk_stim() exists
    - update_rdk_stim() exists
    - dots_stim.draw() exists
    """
    def __init__(self):
        self.update_calls = 0
        self.frame_rate = 60.0

        class _Dots:
            def draw(self_inner):
                return None

        self.dots_stim = _Dots()

    def initialize_rdk_stim(self, direction, coherence):
        # Trial ignores return value; just needs it not to crash
        self.direction = direction
        self.coherence = coherence

    def update_rdk_stim(self):
        self.update_calls += 1


# -----------------------------
# Utility: sync trial clock with win flips
# -----------------------------

@pytest.fixture
def win():
    return DummyWin(dt=1/60)


@pytest.fixture
def rdk():
    return DummyRDK()


def advance_trial_clock_on_flip(win, trial_clock, kb):
    """
    Trial uses:
      - win.callOnFlip(trial_clock.reset)
      - win.flip() timestamps
      - trial_clock.getTime() inside the loop

    To emulate that properly without patching psychopy.core.Clock,
    we advance a DummyClock by dt on each flip.
    """
    original_flip = win.flip

    def flip_wrapped():
        t = original_flip()
        # advance both clocks by the same dt after each flip
        trial_clock._advance(win.dt)
        kb.clock._advance(win.dt)
        return t

    win.flip = flip_wrapped


# ==========================================================
# ✅ TEST 1: Trial ends on response
# ==========================================================

def test_trial_runs_until_response(win, rdk):
    # Keys returned on successive polls:
    # - first poll: no response
    # - second poll: right at rt=0.85
    kb = DummyKb(per_poll_keys=[
        [],
        [KeyEvt("right", 0.85)],
    ])

    trial = Trial(win, kb, rdk, max_stim_sec=2.0, debug=False)

    # Trial creates its own trial_clock internally; we can’t directly inject it,
    # but we *can* rely on the RT coming from kb.clock (earliest.rt).
    # Since we are not asserting trial_clock exact values here, no need to wrap flips.

    result = trial.run_single_trial(direction=0, coherence=1.0)

    assert result["response_key"] == "right"
    assert result["timeout"] == 0
    assert result["reaction_time"] == pytest.approx(0.85)
    assert result["frame_count"] > 0
    assert result["global_onset_time"] is not None
    assert result["response_flip_time"] is not None
    assert result["response_frame_idx"] is not None
    assert rdk.update_calls > 0


# ==========================================================
# ✅ TEST 2: Trial times out with no response
# ==========================================================

def test_trial_times_out(win, rdk):
    # Never return any keys
    kb = DummyKb(per_poll_keys=[])

    # Use a short timeout so the test is fast and deterministic
    trial = Trial(win, kb, rdk, max_stim_sec=0.10, debug=False)

    result = trial.run_single_trial(direction=180, coherence=0.5)

    assert result["response_key"] is None
    assert result["reaction_time"] is None
    assert result["timeout"] == 1

    # Should have a valid onset time and some frames
    assert result["global_onset_time"] is not None
    assert result["frame_count"] > 0

    # Trial clock duration should be at least the timeout (with frame quantization)
    assert result["stimulus_on_screen_duration"] >= 0.10


# ==========================================================
# ✅ TEST 3: Result dict fields are present and consistent
# ==========================================================

def test_trial_result_fields_consistent(win, rdk):
    kb = DummyKb(per_poll_keys=[
        [KeyEvt("left", 0.50)],
    ])

    trial = Trial(win, kb, rdk, max_stim_sec=2.0, debug=False)
    result = trial.run_single_trial(direction=180, coherence=0.5)

    # Required fields in your current Trial implementation
    required = [
        "direction", "coherence",
        "response_key", "reaction_time", "timeout",
        "global_onset_time", "response_flip_time", "response_frame_idx",
        "response_detected_time", "stimulus_on_screen_duration",
        "frame_count", "estimated_fps",
        "frame_stats", "n_long_frames", "max_flip_interval",
    ]
    for k in required:
        assert k in result, f"Missing field: {k}"

    # debug=False -> frame_stats is None (by your code)
    assert result["frame_stats"] is None

    # estimated_fps should be positive (unless duration is 0, which shouldn’t happen)
    assert result["stimulus_on_screen_duration"] > 0
    assert result["estimated_fps"] is not None
    assert result["estimated_fps"] > 0
