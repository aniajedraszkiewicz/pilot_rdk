"""
Geometry validation that reuses Experiment.create_window_and_monitor().
Run with:
    python -m experiment_files.calibration
"""

import sys
import numpy as np
from psychopy import visual, event, gui, core

from .experiment import Experiment


def run_calibration(theta_deg=10.0):
    exp = Experiment()

    # Remove participant field — not needed for calibration
    exp.expInfo.pop("participant", None)

    # Dialog appears on MacBook built-in (unavoidable).
    # Select screen_no = 0 for external monitor, 1 for MacBook built-in.
    dlg = gui.DlgFromDict(exp.expInfo, title="Geometry Calibration")
    if not dlg.OK:
        sys.exit(0)

    # Pass fullscr choice from dialog into exp before creating the window
    exp.fullscr = bool(exp.expInfo.get("fullscr", True))

    # Reuse the experiment's window/monitor logic (Retina handling, sizePix logic, etc.)
    exp.create_window_and_monitor()

    d_cm = float(exp.viewing_distance_cm)
    theta_rad = np.deg2rad(theta_deg)
    expected_cm = 2.0 * d_cm * np.tan(theta_rad / 2.0)

    half = theta_deg / 2.0
    line = visual.Line(
        exp.win,
        start=(-half, 0.0),
        end=(half, 0.0),
        lineWidth=3,
        units="deg",
        color="white",
    )

    msg = visual.TextStim(
        exp.win,
        text=(
            "GEOMETRY CHECK (deg ↔ cm)\n\n"
            f"Line length: {theta_deg:.1f}°\n"
            f"Expected physical length: {expected_cm:.2f} cm\n\n"
            "Measure with a ruler.\n"
            "C = correct, R = incorrect, Q/ESC = quit"
        ),
        pos=(0.0, 4.0),
        height=0.6,
        wrapWidth=24,
        units="deg",
        color="white",
    )

    line.draw()
    msg.draw()
    exp.win.flip()

    key = event.waitKeys(keyList=["c", "r", "q", "escape"])[0]

    exp.win.close()

    if key in ("q", "escape"):
        print("[QUIT] Calibration cancelled.")
    elif key == "c":
        print("[OK] Geometry confirmed.")
    elif key == "r":
        print("[MISMATCH] Geometry mismatch — check screen_width_cm and viewing_distance_cm.")

    sys.exit(0)


if __name__ == "__main__":
    run_calibration(theta_deg=10.0)