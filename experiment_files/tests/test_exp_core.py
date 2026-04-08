"""
Unit tests for the RDK experiment core.

These tests verify that the experiment module:
- imports safely without opening a real PsychoPy window,
- correctly configures display and timing parameters,
- produces deterministic random seeds,
- writes timing and summary files,
- and saves diagnostic outputs.

PsychoPy and OpenGL-dependent components are replaced with lightweight fakes
so the tests can run headlessly and deterministically.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from pathlib import Path
import hashlib

import numpy as np
import pytest


# =========================
# Deterministic helpers
# =========================

@dataclass
class FlipClock:
    dt: float
    now: float = 0.0

    def tick(self) -> float:
        self.now += self.dt
        return self.now


class FakeWindow:
    """Minimal psychopy.visual.Window stand-in (supports refresh-rate test)."""

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.units = kwargs.get("units")
        self.color = kwargs.get("color")

        # experiment.py reads these directly
        self.size = kwargs.get("size", (1440, 900))
        self.useRetina = kwargs.get("useRetina", False)

        # experiment.py uses these in measure_refresh_rate()
        self.recordFrameIntervals = False
        self.frameIntervals = []

        # deterministic flip timing
        self._flip_calls = 0
        self._flip_clock = kwargs.get("_flip_clock", None)

        # deterministic getActualFrameRate()
        self._actual_rate = kwargs.get("_actual_rate", 120.0)

    def getActualFrameRate(self, **_kwargs):
        return self._actual_rate

    def flip(self):
        self._flip_calls += 1

        t = 0.0
        if self._flip_clock is not None:
            t = self._flip_clock.tick()

        # emulate PsychoPy’s interval recording
        if self.recordFrameIntervals:
            if self._flip_calls >= 2 and self._flip_clock is not None:
                self.frameIntervals.append(self._flip_clock.dt)

        return t

    def close(self):
        return None


class FakeMonitor:
    """Minimal psychopy.monitors.Monitor stand-in."""

    def __init__(self, name):
        self.name = name
        self._size_pix = (800, 600)
        self._width_cm = None
        self._distance_cm = None

    def setWidth(self, cm):
        self._width_cm = float(cm)

    def setDistance(self, cm):
        self._distance_cm = float(cm)

    def getWidth(self):
        return float(self._width_cm) if self._width_cm is not None else 0.0

    def getDistance(self):
        return float(self._distance_cm) if self._distance_cm is not None else 0.0

    def setSizePix(self, size_pix):
        self._size_pix = tuple(size_pix)

    def getSizePix(self):
        return self._size_pix

    def save(self):
        return None


class FakeKeyboard:
    """Minimal psychopy.hardware.keyboard.Keyboard stand-in."""

    def __init__(self, backend="iohub"):
        self.backend = backend
        self._cleared = False

    def clearEvents(self):
        self._cleared = True

    # Not needed by current experiment.py tests, but harmless if present later
    def getKeys(self, *args, **kwargs):
        return []


class FakeDlg:
    def __init__(self, ok=True):
        self.OK = ok


class FakeRDK:
    """Replace real RDK during tests (no OpenGL)."""

    def __init__(self, win, frame_rate, dot_speed, dot_density, rng):
        self.win = win
        self.frame_rate = float(frame_rate)
        self.dot_speed = float(dot_speed)
        self.dot_density = float(dot_density)
        self.rng = rng

        # Used by write_summary()
        self.n_dots = 100
        self.field_diameter = 10.0
        self.n_sequences = 3
        self.field_radius = self.field_diameter / 2.0
        self.dots_stim = SimpleNamespace(nElements=self.n_dots)

        # Precomputed geometry values written by write_summary()
        self.spatial_displacement = self.dot_speed / self.frame_rate * self.n_sequences
        self.temporal_displacement = self.n_sequences / self.frame_rate * 1000.0
        self.instantaneous_dot_density = self.dot_density / self.frame_rate


class FakeBlock:
    """Only needed if run_experiment() is tested later."""

    def __init__(self, *args, **kwargs):
        self.block_seed = 999

    def show_intro(self):
        return None

    def show_outro(self):
        return None

    def run_block(self):
        return {
            "threshold_estimates": [0.2, 0.18, 0.17],
            "overall_accuracy": 0.75,
            "quest_start_coh": 0.20,
            "quest_min_coh": 0.05,
            "quest_max_coh": 0.50,
            "quest_start_intensity_log10": -0.70,
            "quest_start_intensity_sd_log10": 0.30,
            "quest_pThreshold": 0.75,
            "quest_gamma": 0.5,
            "quest_beta": 3.5,
            "quest_delta": 0.02,
            "quest_nTrials": 64,
            "quest_method": "quantile",
            "coh_min_used": 0.05,
            "coh_max_used": 0.50,
            "mean_coh": 0.17,
            "ci_5_95_coh": (0.10, 0.25),
            "ci_5_95_log10": (-1.0, -0.6),
            "last10_accuracy": 0.8,
            "bias_sensitivity_right": 0.75,
            "bias_specificity_right": 0.80,
        }


# =========================
# Fixture factory: safe import
# =========================

def _import_experiment_module(monkeypatch, backend: str):
    """
    Import experiment_files.experiment safely by stubbing:
    - psychopy and its submodules
    - experiment_files.block and experiment_files.rdk_stim
    - os.chdir at import-time

    The keyboard backend is selected via RDK_KEYBOARD_BACKEND.
    """
    # Prevent import-time os.chdir(...) from changing pytest cwd
    monkeypatch.setattr(os, "chdir", lambda *_a, **_k: None)

    # Select backend for this import
    monkeypatch.setenv("RDK_KEYBOARD_BACKEND", backend)

    # Ensure project root is importable so "experiment_files" can be found
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Ensure parent package exists for submodule stubs
    if "experiment_files" not in sys.modules:
        pkg = ModuleType("experiment_files")
        pkg.__path__ = [os.path.join(project_root, "experiment_files")]
        monkeypatch.setitem(sys.modules, "experiment_files", pkg)

    # -------- Fake psychopy tree --------
    psychopy = ModuleType("psychopy")
    psychopy.__version__ = "FAKE-PSYCHOPY"

    prefs = SimpleNamespace(general={}, hardware={})
    psychopy.prefs = prefs

    core = SimpleNamespace(quit=lambda: (_ for _ in ()).throw(SystemExit("core.quit called")))
    psychopy.core = core

    gui = SimpleNamespace(DlgFromDict=lambda *a, **k: FakeDlg(ok=True))
    psychopy.gui = gui

    monitors = SimpleNamespace(Monitor=lambda name: FakeMonitor(name))
    psychopy.monitors = monitors

    visual = SimpleNamespace(
        Window=FakeWindow,
        TextStim=lambda *a, **k: SimpleNamespace(draw=lambda: None),
    )
    psychopy.visual = visual

    data = SimpleNamespace()
    psychopy.data = data

    keyboard_mod = SimpleNamespace(Keyboard=FakeKeyboard)
    hardware = SimpleNamespace(keyboard=keyboard_mod)
    psychopy.hardware = hardware

    monkeypatch.setitem(sys.modules, "psychopy", psychopy)
    monkeypatch.setitem(sys.modules, "psychopy.prefs", prefs)
    monkeypatch.setitem(sys.modules, "psychopy.core", core)
    monkeypatch.setitem(sys.modules, "psychopy.gui", gui)
    monkeypatch.setitem(sys.modules, "psychopy.monitors", monitors)
    monkeypatch.setitem(sys.modules, "psychopy.visual", visual)
    monkeypatch.setitem(sys.modules, "psychopy.data", data)
    monkeypatch.setitem(sys.modules, "psychopy.hardware", hardware)
    monkeypatch.setitem(sys.modules, "psychopy.hardware.keyboard", keyboard_mod)

    # -------- Stub internal modules imported at top-level --------
    fake_block_mod = ModuleType("experiment_files.block")
    fake_block_mod.Block = FakeBlock
    monkeypatch.setitem(sys.modules, "experiment_files.block", fake_block_mod)

    fake_rdk_mod = ModuleType("experiment_files.rdk_stim")
    fake_rdk_mod.RDK = FakeRDK
    monkeypatch.setitem(sys.modules, "experiment_files.rdk_stim", fake_rdk_mod)

    # Import safely
    mod_name = "experiment_files.experiment"
    if mod_name in sys.modules:
        mod = importlib.reload(sys.modules[mod_name])
    else:
        mod = importlib.import_module(mod_name)

    return mod


@pytest.fixture(params=["iohub", "ptb"])
def exp_mod(request, monkeypatch):
    """Parametrized experiment module fixture for both keyboard backends."""
    return _import_experiment_module(monkeypatch, request.param)


# =========================
# Tests
# =========================

def test_import_sets_expected_prefs_and_env(exp_mod):
    backend = exp_mod.PREFERRED_KEYBOARD_BACKEND

    if backend == "iohub":
        assert os.environ.get("PSYCHOPY_USE_IOHUB") == "True"
        assert os.environ.get("PSYCHOPY_NO_PTBOXLIB") == "1"
    elif backend == "ptb":
        assert os.environ.get("PSYCHOPY_USE_IOHUB") == "False"
        assert os.environ.get("PSYCHOPY_NO_PTBOXLIB") == "0"
    else:
        pytest.fail(f"Unexpected backend: {backend}")

    assert exp_mod.prefs.general.get("winType") == "pyglet"
    assert exp_mod.prefs.general.get("waitBlanking") is True
    assert exp_mod.prefs.hardware.get("keyboard") == backend


def test_create_window_and_monitor_sets_window_fields(exp_mod):
    exp = exp_mod.Experiment()
    exp.expInfo["screen_width_cm"] = "53.0"
    exp.expInfo["viewing_distance_cm"] = "57.0"
    exp.expInfo["fullscr"] = False

    # create_window_and_monitor() reads self.fullscr, normally set in collect_participant_info()
    exp.fullscr = False

    exp.create_window_and_monitor()

    assert exp.win is not None
    assert exp.win.units == "deg"
    assert exp.win.kwargs["fullscr"] is False
    assert exp.win.kwargs["waitBlanking"] is True
    assert exp.win.kwargs["useFBO"] is True


def test_measure_refresh_rate_deterministic_and_writes_files(tmp_path, exp_mod):
    exp = exp_mod.Experiment()
    exp.results_csv_path = str(tmp_path / "S01_20250101_000000.csv")

    clk = FlipClock(dt=1.0 / 120.0)
    exp.win = FakeWindow(_flip_clock=clk, _actual_rate=120.0)

    measured = exp.measure_refresh_rate()

    assert np.isfinite(measured)
    assert abs(measured - 120.0) < 1.0
    assert os.path.exists(exp.frame_times_path)
    assert os.path.exists(exp.frame_intervals_path)


def test_measure_and_define_parameters_sets_speed_and_density(exp_mod, monkeypatch):
    exp = exp_mod.Experiment()
    monkeypatch.setattr(exp, "measure_refresh_rate", lambda: 120.0, raising=True)

    exp.measure_and_define_parameters()

    assert abs(exp.measured_rate - 120.0) < 1e-12
    assert abs(exp.dot_speed - (0.28 * 120.0 / 3.0)) < 1e-12
    assert abs(exp.dot_density - 24.0) < 1e-12


def test_initialize_stimulus_creates_deterministic_seed_and_rdk(tmp_path, exp_mod, monkeypatch):
    exp = exp_mod.Experiment()

    exp.subject_id = "S01"
    exp.run_id = "20250101_000000_abcd"
    exp.measured_rate = 120.0
    exp.dot_speed = 5.0
    exp.dot_density = 0.55
    exp.results_csv_path = str(tmp_path / "S01_20250101_000000.csv")
    exp.win = FakeWindow()

    called = {"n": 0}
    monkeypatch.setattr(
        exp, "write_summary",
        lambda: called.__setitem__("n", called["n"] + 1),
        raising=True
    )

    exp.initialize_stimulus_and_load_trials()

    base = f"RDK|{exp.subject_id}|{exp.run_id}".encode("utf-8")
    digest = hashlib.sha256(base).hexdigest()
    expected_seed = int(digest[:8], 16)

    assert exp.kb.backend == exp.keyboard_backend_preferred
    assert exp.keyboard_backend_used == exp.keyboard_backend_preferred
    assert exp.kb._cleared is True
    assert exp.rdk_seed == expected_seed
    assert exp.rdk.frame_rate == 120.0
    assert called["n"] == 1


def test_write_summary_creates_file_and_contains_expected_fields(tmp_path, exp_mod):
    exp = exp_mod.Experiment()

    exp.subject_id = "S01"
    exp.run_id = "20250101_000000_abcd"
    exp.results_csv_path = str(tmp_path / "S01_20250101_000000.csv")

    # geometry fields
    exp.screen_width_cm = 53.0
    exp.viewing_distance_cm = 57.0
    exp.actual_win_size_px = (1440, 900)
    exp.monitor_model_size_px = (1440, 900)

    # timing-check section expects these paths (basenames are written)
    exp.frame_times_path = str(tmp_path / "S01_20250101_000000_refresh_timestamps.txt")
    exp.frame_intervals_path = str(tmp_path / "S01_20250101_000000_refresh_intervals.txt")

    # stimulus/timing fields written by write_summary()
    exp.measured_rate = 120.0
    exp.refresh_rate_method = "hardcoded"
    exp.dot_speed = 5.0
    exp.dot_density = 0.55
    exp.keyboard_backend_used = exp.keyboard_backend_preferred

    exp.rdk = FakeRDK(
        FakeWindow(),
        exp.measured_rate,
        exp.dot_speed,
        exp.dot_density,
        np.random.default_rng(0),
    )

    exp.write_summary()

    summary_path = exp.results_csv_path.replace(".csv", "_summary.txt")
    assert os.path.exists(summary_path)

    txt = Path(summary_path).read_text(encoding="utf-8")
    assert "RUN SUMMARY" in txt
    assert "DISPLAY TIMING" in txt
    assert "DISPLAY TIMING CHECK" in txt
    assert "RDK STIMULUS" in txt
    assert "ENVIRONMENT" in txt
    assert "Run ID:" in txt
    assert "Hardcoded refresh rate used (Hz):" in txt
    assert "PsychoPy version:" in txt
    assert "Screen width (cm):" in txt
    assert "Viewing distance (cm):" in txt
    assert "Keyboard backend preferred:" in txt
    assert "Keyboard backend used:" in txt


def test_plot_diagnostics_saves_png(tmp_path, exp_mod):
    diagnostics = {"threshold_estimates": [0.3, 0.25, 0.22, 0.21]}
    base = str(tmp_path / "diag_test")

    exp_mod.plot_diagnostics(diagnostics, base)

    out = base + "_quest_diagnostics.png"
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0