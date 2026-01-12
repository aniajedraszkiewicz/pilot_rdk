"""
test_exp_core.py

Headless-safe unit tests for experiment_files/experiment.py.

Key idea:
- experiment.py imports .block and .rdk_stim at import-time
- block.py imports psychopy.data (and other PsychoPy stuff)
So in tests we must:
1) Provide a fake psychopy package that includes psychopy.data
2) Provide fake experiment_files.block and experiment_files.rdk_stim modules
   BEFORE importing experiment_files.experiment, so real block.py never loads.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

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
    """Minimal psychopy.visual.Window stand-in."""
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.units = kwargs.get("units")
        self.color = kwargs.get("color")
        self.size = (1440, 900)  # default fake size
        self._flip_calls = 0
        self._flip_clock = kwargs.get("_flip_clock", None)

    def flip(self):
        self._flip_calls += 1
        if self._flip_clock is not None:
            return self._flip_clock.tick()
        return 0.0

    def close(self):
        return None


class FakeMonitor:
    """Minimal psychopy.monitors.Monitor stand-in."""
    def __init__(self, name):
        self.name = name
        self._size_pix = (1440, 900)

    def setWidth(self, _cm): ...
    def setDistance(self, _cm): ...
    def setSizePix(self, size_pix): self._size_pix = tuple(size_pix)
    def getSizePix(self): return self._size_pix
    def save(self): return None


class FakeKeyboard:
    """Minimal psychopy.hardware.keyboard.Keyboard stand-in."""
    def __init__(self, backend="iohub"):
        self.backend = backend
        self._cleared = False

    def clearEvents(self):
        self._cleared = True


class FakeDlg:
    def __init__(self, ok=True):
        self.OK = ok


class FakeRDK:
    """Replace your real RDK during tests (no ElementArrayStim, no OpenGL)."""
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
        self.dots_stim = SimpleNamespace(nElements=self.n_dots)


class FakeBlock:
    """Only needed if you ever test run_experiment() later."""
    def __init__(self, *args, **kwargs):
        self.block_seed = 999

    def show_intro(self): return None

    def run_block(self):
        return {
            "threshold_estimates": [0.2, 0.18, 0.17],
            "overall_accuracy": 0.75,
            "quest_first_intensity": 0.2,
            "quest_intensity_min": 0.05,
            "quest_intensity_max": 0.5,
            "mean": 0.17,
            "sd": 0.02,
            "mode": 0.17,
            "ci_5_95": (0.10, 0.25),
            "last10_accuracy": 0.8,
        }


def install_fake_psychopy(monkeypatch):
    """Install a minimal fake psychopy module tree into sys.modules."""
    psychopy = ModuleType("psychopy")
    psychopy.__version__ = "FAKE-PSYCHOPY"

    prefs = SimpleNamespace(general={}, hardware={})
    psychopy.prefs = prefs

    core = SimpleNamespace(quit=lambda: (_ for _ in ()).throw(SystemExit("core.quit called")))
    psychopy.core = core

    gui = SimpleNamespace(DlgFromDict=lambda *a, **k: FakeDlg(ok=True))
    psychopy.gui = gui

    monitors = SimpleNamespace(
        getAllMonitors=lambda: ["MacBookDisplay"],
        Monitor=lambda name: FakeMonitor(name),
    )
    psychopy.monitors = monitors

    visual = SimpleNamespace(Window=FakeWindow)
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

    return psychopy


# =========================
# Fixture: safe import
# =========================

@pytest.fixture()
def exp_mod(monkeypatch):
    """
    Import experiment_files.experiment safely by stubbing:
    - psychopy and its submodules (including psychopy.data)
    - experiment_files.block and experiment_files.rdk_stim (so real files aren't imported)
    - os.chdir at import-time (so tests don't change cwd)
    """
    # Prevent import-time os.chdir(...) from changing pytest cwd
    monkeypatch.setattr(os, "chdir", lambda *_a, **_k: None)

    # -------- Fake psychopy tree --------
    install_fake_psychopy(monkeypatch)

    # -------- Stub your internal modules imported at top-level --------
    # experiment.py does: from .block import Block ; from .rdk_stim import RDK
    fake_block_mod = ModuleType("experiment_files.block")
    fake_block_mod.Block = FakeBlock
    monkeypatch.setitem(sys.modules, "experiment_files.block", fake_block_mod)

    fake_rdk_mod = ModuleType("experiment_files.rdk_stim")
    fake_rdk_mod.RDK = FakeRDK
    monkeypatch.setitem(sys.modules, "experiment_files.rdk_stim", fake_rdk_mod)

    # Now we can import experiment_files.experiment safely
    mod_name = "experiment_files.experiment"
    if mod_name in sys.modules:
        mod = importlib.reload(sys.modules[mod_name])
    else:
        mod = importlib.import_module(mod_name)

    return mod


# =========================
# Tests
# =========================

def test_import_sets_expected_prefs_and_env(exp_mod):
    assert os.environ.get("PSYCHOPY_USE_IOHUB") == "True"
    assert os.environ.get("PSYCHOPY_NO_PTBOXLIB") == "1"

    assert exp_mod.prefs.general.get("winType") == "pyglet"
    assert exp_mod.prefs.general.get("waitBlanking") is True
    assert exp_mod.prefs.hardware.get("keyboard") == "iohub"


def test_create_window_and_monitor_warms_up_flips(exp_mod):
    exp = exp_mod.Experiment()
    # Use "Custom" to avoid depending on detected_display_label which changes per system
    exp.monitor_choice = "Custom"

    exp.create_window_and_monitor()

    assert exp.win is not None
    assert exp.win.units == "deg"
    assert exp.win.kwargs["fullscr"] is True
    assert exp.win.kwargs["waitBlanking"] is True
    assert exp.win.kwargs["useFBO"] is True
    assert exp.win._flip_calls >= 120


def test_measure_refresh_rate_deterministic(exp_mod):
    exp = exp_mod.Experiment()

    clk = FlipClock(dt=1.0 / 120.0)
    exp.win = FakeWindow(_flip_clock=clk)

    measured = exp.measure_refresh_rate()
    assert np.isfinite(measured)
    assert abs(measured - 120.0) < 1.0


def test_density_correction_keeps_dots_per_frame_stable(exp_mod, monkeypatch):
    exp = exp_mod.Experiment()

    monkeypatch.setattr(exp, "measure_refresh_rate", lambda: 120.0, raising=True)
    exp.measure_and_define_parameters()

    assert abs(exp.dots_per_frame_adjusted - exp.dots_per_frame_baseline) < 1e-12


def test_initialize_stimulus_creates_seed_and_rdk(tmp_path, exp_mod, monkeypatch):
    exp = exp_mod.Experiment()

    exp.subject_id = "S01"
    exp.monitor_choice = "MacBookDisplay"
    exp.measured_rate = 120.0
    exp.dot_speed = 5.0
    exp.density_adjusted = 33.4
    exp.results_csv_path = str(tmp_path / "S01_20250101_000000.csv")
    exp.win = FakeWindow()

    # Don’t test file I/O here; just confirm it is called
    called = {"n": 0}
    monkeypatch.setattr(exp, "write_summary", lambda: called.__setitem__("n", called["n"] + 1), raising=True)

    exp.initialize_stimulus_and_load_trials()

    assert exp.kb.backend == "iohub"
    assert exp.kb._cleared is True

    assert isinstance(exp.rdk_seed, int)
    assert 0 <= exp.rdk_seed <= 0xFFFFFFFF

    assert exp.rdk.frame_rate == 120.0
    assert called["n"] == 1


def test_write_summary_creates_file_and_contains_expected_fields(tmp_path, exp_mod):
    exp = exp_mod.Experiment()

    exp.subject_id = "S01"
    exp.monitor_choice = "Custom"
    exp.results_csv_path = str(tmp_path / "S01_20250101_000000.csv")

    # These are set by create_window_and_monitor but we need them for write_summary
    exp.screen_width_cm = 53.0
    exp.viewing_distance_cm = 60.0
    exp.resolution_x_px = 1920
    exp.resolution_y_px = 1080

    exp.measured_rate = 120.0
    exp.DESIGN_RATE = 60.0
    exp.dot_speed = 5.0
    exp.dot_density = 16.7
    exp.density_adjusted = exp.dot_density * (exp.measured_rate / exp.DESIGN_RATE)
    exp.dots_per_frame_baseline = exp.dot_density / exp.DESIGN_RATE
    exp.dots_per_frame_adjusted = exp.density_adjusted / exp.measured_rate

    exp.rdk = FakeRDK(FakeWindow(), exp.measured_rate, exp.dot_speed, exp.density_adjusted, np.random.default_rng(0))

    exp.write_summary()

    summary_path = exp.results_csv_path.replace(".csv", "_summary.txt")
    assert os.path.exists(summary_path)

    txt = open(summary_path, "r", encoding="utf-8").read()
    assert "RUN SUMMARY" in txt
    assert "DISPLAY TIMING" in txt
    assert "RDK STIMULUS" in txt
    assert "ENVIRONMENT" in txt
    assert "Measured refresh rate (Hz):" in txt
    assert "PsychoPy version:" in txt


def test_plot_diagnostics_saves_png(tmp_path, exp_mod):
    diagnostics = {"threshold_estimates": [0.3, 0.25, 0.22, 0.21]}
    base = str(tmp_path / "diag_test")

    exp_mod.plot_diagnostics(diagnostics, base)

    out = base + "_thresholds.png"
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_auto_resolution_defaults_use_detected_screen(monkeypatch):
    # Ensure a clean import of experiment_files.experiment
    mod_name = "experiment_files.experiment"
    for name in [mod_name, "experiment_files.block", "experiment_files.rdk_stim", "tkinter"]:
        sys.modules.pop(name, None)

    # Do not change cwd during import
    monkeypatch.setattr(os, "chdir", lambda *_a, **_k: None)

    # Fake tkinter with predictable screen dimensions
    class _FakeTk:
        def withdraw(self):
            return None
        def winfo_screenwidth(self):
            return 1600
        def winfo_screenheight(self):
            return 900
        def destroy(self):
            return None

    fake_tk = SimpleNamespace(Tk=_FakeTk)
    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)

    # Install fake psychopy tree and internal modules
    install_fake_psychopy(monkeypatch)

    fake_block_mod = ModuleType("experiment_files.block")
    fake_block_mod.Block = FakeBlock
    monkeypatch.setitem(sys.modules, "experiment_files.block", fake_block_mod)

    fake_rdk_mod = ModuleType("experiment_files.rdk_stim")
    fake_rdk_mod.RDK = FakeRDK
    monkeypatch.setitem(sys.modules, "experiment_files.rdk_stim", fake_rdk_mod)

    mod = importlib.import_module(mod_name)

    assert mod.screen_width_px == 1600
    assert mod.screen_height_px == 900
    assert mod.detected_display_label == "Auto (1600x900)"

    exp = mod.Experiment()
    assert exp.expInfo["monitor"][0] == mod.detected_display_label
    assert exp.expInfo["resolution_x_px"] == "1600"
    assert exp.expInfo["resolution_y_px"] == "900"

    # Clean up for subsequent tests
    sys.modules.pop(mod_name, None)
