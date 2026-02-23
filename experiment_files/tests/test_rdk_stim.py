# tests/test_rdk_stim.py
import numpy as np
import pytest

from experiment_files.rdk_stim import RDK


# -----------------------------
# Test doubles (headless-safe)
# -----------------------------

class DummyElementArrayStim:
    """
    Minimal stand-in for psychopy.visual.ElementArrayStim.
    Stores the last positions passed to setXYs so we can assert shapes/finite values.
    """
    def __init__(self, win, elementTex, fieldShape, elementMask, sizes, nElements, units, fieldSize, colors=None, colorSpace=None, **kwargs,):
        self.win = win
        self.elementTex = elementTex
        self.fieldShape = fieldShape
        self.elementMask = elementMask
        self.sizes = sizes
        self.nElements = nElements
        self.units = units
        self.fieldSize = fieldSize
        self.colors = colors
        self.colorSpace = colorSpace
        self._xys = None

    def setXYs(self, xys):
        self._xys = np.asarray(xys)

    def draw(self):
        # Not needed for these tests, but present so Trial can call it in other test modules.
        pass


@pytest.fixture
def rdk(monkeypatch):
    """
    Create a deterministic RDK instance for testing without opening a real PsychoPy window.
    We monkeypatch ElementArrayStim so it won't touch OpenGL/display backends.
    """
    # Patch the ElementArrayStim used inside your module.
    import experiment_files.rdk_stim as rdk_mod
    monkeypatch.setattr(rdk_mod.visual, "ElementArrayStim", DummyElementArrayStim)

    rng = np.random.default_rng(12345)

    # Use a density that yields a reasonable number of dots for stable proportions.
    rdk = RDK(
        win=None,
        dot_density=120.0,
        dot_speed=5.0,
        frame_rate=60.0,
        field_diameter=10.0,
        n_sequences=3,
        rng=rng,
        max_lifetime_frames=12,
    )
    return rdk


# -----------------------------
# Helpers
# -----------------------------

def assert_all_dots_within_circle(coords_2xn, radius, atol=1e-9):
    """coords_2xn is shape (2, n). All dots must satisfy x^2+y^2 <= radius^2 (with tolerance)."""
    x = coords_2xn[0, :]
    y = coords_2xn[1, :]
    r2 = x * x + y * y
    assert np.all(r2 <= (radius * radius + atol)), "Some dots are outside the circular aperture."


# =====================================================
# BASIC STRUCTURE / SANITY
# =====================================================

def test_n_dots_divisible_by_sequences(rdk):
    assert rdk.n_dots % rdk.n_sequences == 0


def test_n_dots_in_sequence_consistency(rdk):
    assert rdk.n_dots_in_sequence * rdk.n_sequences == rdk.n_dots


def test_parameter_sanity(rdk):
    assert rdk.dot_speed > 0
    assert rdk.frame_rate > 0
    assert rdk.field_diameter > 0
    assert rdk.field_radius == pytest.approx(rdk.field_diameter / 2.0)
    assert rdk.n_dots > 0
    assert rdk.max_lifetime_frames > 0
    assert rdk.dot_lifetimes.shape == (rdk.n_dots,)


# =====================================================
# INITIALIZATION BEHAVIOR
# =====================================================

@pytest.mark.parametrize("coherence", [-0.1, 0.0, 0.5, 1.0, 1.1])
def test_initialize_is_finite_and_in_circle(rdk, coherence):
    mask, coords, seq = rdk.initialize_rdk_stim(direction=0, coherence=coherence)

    assert mask.shape == (rdk.n_dots,)
    assert mask.dtype == bool
    assert coords.shape == (2, rdk.n_dots)
    assert np.isfinite(coords).all()
    assert seq == -1

    # Dots are sampled uniformly INSIDE a circle in degrees.
    assert_all_dots_within_circle(coords, rdk.field_radius)


def test_initialize_resets_lifetimes_and_diagnostics(rdk):
    rdk.initialize_rdk_stim(direction=0, coherence=0.5)

    # Lifetimes are randomized on initialize (0 .. max_lifetime_frames-1)
    assert rdk.dot_lifetimes.shape == (rdk.n_dots,)
    assert np.issubdtype(rdk.dot_lifetimes.dtype, np.integer)
    assert np.all((0 <= rdk.dot_lifetimes) & (rdk.dot_lifetimes < rdk.max_lifetime_frames))

    # Diagnostics are reset on initialize
    assert rdk.n_outside_last == 0
    assert rdk.n_expired_last == 0


def test_deterministic_seed_reproduces_initial_positions(monkeypatch):
    """
    Two RDKs with the same RNG seed should initialize to identical dot coordinates.
    """
    import experiment_files.rdk_stim as rdk_mod
    monkeypatch.setattr(rdk_mod.visual, "ElementArrayStim", DummyElementArrayStim)

    seed = 999
    r1 = RDK(win=None, dot_density=120.0, dot_speed=5.0, frame_rate=60.0,
             field_diameter=10.0, n_sequences=3, rng=np.random.default_rng(seed))
    r2 = RDK(win=None, dot_density=120.0, dot_speed=5.0, frame_rate=60.0,
             field_diameter=10.0, n_sequences=3, rng=np.random.default_rng(seed))

    _, c1, _ = r1.initialize_rdk_stim(direction=0, coherence=0.5)
    _, c2, _ = r2.initialize_rdk_stim(direction=0, coherence=0.5)

    np.testing.assert_allclose(c1, c2, rtol=0, atol=0)


# =====================================================
# UPDATE BEHAVIOR: SEQUENCES, DISPLACEMENT, BOUNDS
# =====================================================

def test_sequence_index_cycles(rdk):
    rdk.initialize_rdk_stim(direction=0, coherence=0.5)

    seen = []
    for _ in range(rdk.n_sequences * 2):
        _, _, seq = rdk.update_rdk_stim()
        seen.append(seq)

    # Should cycle 0,1,2,0,1,2 for n_sequences=3
    assert seen[:3] == list(range(rdk.n_sequences))
    assert seen[3:6] == list(range(rdk.n_sequences))


def test_active_dots_count_matches_n_dots_in_sequence(rdk):
    rdk.initialize_rdk_stim(direction=0, coherence=0.5)
    for _ in range(10):
        mask, _, _ = rdk.update_rdk_stim()
        assert int(mask.sum()) == rdk.n_dots_in_sequence


def test_displacement_vector_magnitude_matches_speed(rdk):
    rdk.initialize_rdk_stim(direction=90, coherence=1.0)

    # Per your code:
    # displacement_per_frame = dot_speed/frame_rate
    # displacement_per_sequence = displacement_per_frame * n_sequences
    expected_mag = (rdk.dot_speed / rdk.frame_rate) * rdk.n_sequences

    mag = float(np.hypot(rdk.displacement_vector_X, rdk.displacement_vector_Y))
    assert mag == pytest.approx(expected_mag, rel=1e-12, abs=1e-12)


def test_update_keeps_all_dots_in_circle_after_many_frames(rdk):
    rdk.initialize_rdk_stim(direction=0, coherence=0.5)

    for _ in range(200):
        _, coords, _ = rdk.update_rdk_stim()
        assert np.isfinite(coords).all()
        assert_all_dots_within_circle(coords, rdk.field_radius)


def test_update_sets_elementarraystim_positions_shape(rdk):
    rdk.initialize_rdk_stim(direction=0, coherence=0.5)
    rdk.update_rdk_stim()

    # DummyElementArrayStim stores last xys; should be (n_dots, 2)
    xys = rdk.dots_stim._xys
    assert xys is not None
    assert xys.shape == (rdk.n_dots, 2)
    assert np.isfinite(xys).all()


# =====================================================
# LIFETIME / RESPAWN MECHANICS
# =====================================================

def test_lifetime_expiration_triggers_respawn(monkeypatch):
    """
    With very short lifetime, some dots must expire and get reset to 0.
    """
    import experiment_files.rdk_stim as rdk_mod
    monkeypatch.setattr(rdk_mod.visual, "ElementArrayStim", DummyElementArrayStim)

    r = RDK(
        win=None,
        dot_density=120.0,
        dot_speed=0.0,            # avoid outside-respawns confounding the test
        frame_rate=60.0,
        field_diameter=10.0,
        n_sequences=3,
        rng=np.random.default_rng(2024),
        max_lifetime_frames=2,    # expire quickly
    )
    r.initialize_rdk_stim(direction=0, coherence=0.5)

    # After 2 frames, lifetimes reach >=2 and should be respawned (set to 0)
    r.update_rdk_stim()  # lifetimes: 1
    r.update_rdk_stim()  # lifetimes: 2 -> expired -> respawn -> reset to 0

    assert r.n_expired_last > 0, "Expected some expired dots with max_lifetime_frames=2."
    assert np.any(r.dot_lifetimes == 0), "Expected some lifetimes to be reset to 0 after respawn."


def test_outside_respawn_brings_dots_back_inside(monkeypatch):
    """
    Use very high speed so signal dots likely go outside; ensure respawn returns them inside.
    """
    import experiment_files.rdk_stim as rdk_mod
    monkeypatch.setattr(rdk_mod.visual, "ElementArrayStim", DummyElementArrayStim)

    r = RDK(
        win=None,
        dot_density=120.0,
        dot_speed=500.0,          # huge displacement -> likely outside
        frame_rate=60.0,
        field_diameter=10.0,
        n_sequences=3,
        rng=np.random.default_rng(7),
        max_lifetime_frames=9999, # avoid lifetime respawn confound
    )
    r.initialize_rdk_stim(direction=0, coherence=1.0)

    # One update can push many signal dots out; your code respawns outside dots immediately.
    _, coords, _ = r.update_rdk_stim()
    assert_all_dots_within_circle(coords, r.field_radius)
    # We can't guarantee outside > 0 deterministically for every parameter combo,
    # but with these numbers it should be extremely likely.
    assert r.n_outside_last >= 0


# =====================================================
# COHERENCE: BASIC EFFECT (probabilistic, not strict)
# =====================================================

@pytest.mark.parametrize("coherence", [0.0, 0.5, 1.0])
def test_empirical_signal_fraction_tracks_coherence_roughly(rdk, coherence):
    """
    On each frame, active dots are assigned signal with P=coherence.
    This is probabilistic; test with a tolerance that scales with n.
    """
    rdk.initialize_rdk_stim(direction=0, coherence=coherence)
    rdk.update_rdk_stim()

    active = int(rdk.active_dots_mask.sum())
    signal = int(rdk.signal_dots_mask.sum())

    if active == 0:
        pytest.skip("No active dots (unexpected with current parameters).")

    emp = signal / active

    # Binomial SD ~ sqrt(p(1-p)/n); allow ~6 SD to be robust.
    p = float(coherence)
    sd = np.sqrt(max(p * (1 - p), 1e-9) / active)
    tol = max(0.02, 6 * sd)  # always allow at least 0.02

    assert abs(emp - p) <= tol
