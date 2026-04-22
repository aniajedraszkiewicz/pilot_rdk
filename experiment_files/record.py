import numpy as np
from psychopy import visual, core, event, monitors

from experiment_files.rdk_stim import RDK


# ── CHANGED: all parameters now match experiment.py exactly ─────────────────
HARDCODED_REFRESH_RATE_HZ = 120.0
SCREEN_WIDTH_CM           = 52.4
VIEWING_DISTANCE_CM       = 57.0
SCREEN_NO                 = 0
SCREEN_RES                = (1920, 1080)
TARGET_DISPLACEMENT       = 0.28
DOT_SPEED                 = TARGET_DISPLACEMENT * HARDCODED_REFRESH_RATE_HZ / 3.0
DOT_DENSITY               = 24.0
FIELD_DIAMETER            = 18.0
N_SEQUENCES               = 3
MAX_LIFETIME_FRAMES       = 36
DOT_SIZE_DEG              = 0.11

def measure_refresh_rate(win):
    """Measure actual refresh rate if possible; fall back to 120 Hz."""
    hz = win.getActualFrameRate(nIdentical=10, nMaxFrames=240, nWarmUpFrames=30, threshold=1.0)
    if hz is None or (not np.isfinite(hz)) or hz <= 0:
        hz = 120.0
    return float(hz)


def run_demo_for_coherence(win, rdk, coh, direction_deg, seconds, label, fps):
    """Show one coherence level for a fixed duration (no frame recording)."""

    # --------------------
    # Interval: blank screen with coherence label
    # --------------------
    label.height = 2.0
    label.pos = (0, 0)
    label.text = f"Coherence = {coh:.2f}"
    n_frames_interval = int(round(0.5 * fps))  # 0.5 s

    for _ in range(n_frames_interval):
        if "escape" in event.getKeys():
            win.close()
            core.quit()
        label.draw()
        win.flip()

    rdk.initialize_rdk_stim(direction=direction_deg, coherence=float(coh))

    speed_per_update = (rdk.displacement_vector_X**2 + rdk.displacement_vector_Y**2) ** 0.5
    print(f"[coh={coh:.2f}] disp/update = {speed_per_update:.4f} deg")

    label.height = 0.8
    label.pos = (0, -8)
    label.text = f"Coherence = {coh:.2f}    Direction = {direction_deg:.0f}°"

    n_frames = int(round(seconds * fps))

    printed = False
    emp_vals = []

    for frame_i in range(n_frames):
        if "escape" in event.getKeys():
            win.close()
            core.quit()

        rdk.update_rdk_stim()

        # First-frame print
        if (not printed) and rdk.current_dots_count > 0:
            emp_coh = rdk.n_signal_dots / rdk.current_dots_count
            print(
                f"[coh={coh:.2f}] "
                f"frame={frame_i} | "
                f"empirical={emp_coh:.3f} | "
                f"nSig={rdk.n_signal_dots}/{rdk.current_dots_count}"
            )
            printed = True

        # Collect empirical coherence
        if rdk.current_dots_count > 0:
            emp_vals.append(rdk.n_signal_dots / rdk.current_dots_count)

        rdk.dots_stim.draw()
        label.draw()
        win.flip()

    # Mean empirical coherence across all frames
    if len(emp_vals) > 0:
        print(
            f"[coh={coh:.2f}] "
            f"empirical_mean={np.mean(emp_vals):.3f} | "
            f"empirical_sd={np.std(emp_vals):.3f} | "
            f"n_frames={len(emp_vals)}"
        )
    else:
        print(f"[coh={coh:.2f}] empirical_mean=NA (no frames with valid dot count)")


def main():
    # --------------------
    # Window
    # --------------------
    mon = monitors.Monitor("testMonitor")
    mon.setWidth(34.5)          # cm
    mon.setDistance(57)         # cm
    mon.setSizePix((1200, 800))

    win = visual.Window(
        size=(1200, 800),
        units="deg",
        monitor=mon,
        fullscr=True,
        color="black",
        waitBlanking=True,
        checkTiming=False
    )

    fps = measure_refresh_rate(win)
    print(f"Measured refresh rate: {fps:.2f} Hz")

    # --------------------
    # Stimulus
    # --------------------
    rdk = RDK(
        win=win,
        dot_density=25.0,
        dot_speed=5.0,
        frame_rate=fps,          # match measured (or fallback) fps for correct speed
        field_diameter=18.0,
        n_sequences=3,
        rng=None,
        max_lifetime_frames=36
    )

    label = visual.TextStim(
        win,
        text="",
        pos=(0, -8),
        height=0.8,
        color="white"
    )

    # --------------------
    # Demo settings
    # --------------------
    coherences = np.round(np.arange(0.1, 1.01, 0.1), 2)
    direction_deg = 0
    seconds_per_level = 3.0

    # --------------------
    # Run demo
    # --------------------
    for coh in coherences:
        run_demo_for_coherence(
            win=win,
            rdk=rdk,
            coh=coh,
            direction_deg=direction_deg,
            seconds=seconds_per_level,
            label=label,
            fps=fps
        )

    win.close()
    core.quit()


if __name__ == "__main__":
    main()
