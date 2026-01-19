"""
Geometry validation that reuses Experiment.create_window_and_monitor().
Run with:
    python -m experiment_files.calibration
"""

import numpy as np
from psychopy import visual, core, event, gui

from .experiment import Experiment


def run_calibration(theta_deg=10.0, fullscr=True):
    exp = Experiment()

    # Override defaults for this machine (optional)
    exp.expInfo["screen_width_cm"] = "31.0"
    exp.expInfo["viewing_distance_cm"] = "57.0"
    exp.expInfo["fullscr"] = True  # keep simple default for GUI

    # Show the same-style dialog as the experiment
    dlg = gui.DlgFromDict(exp.expInfo, title="Geometry Calibration")
    if not dlg.OK:
        core.quit()

    # Match experiment semantics: create_window_and_monitor reads exp.expInfo + exp.fullscr
    exp.fullscr = bool(exp.expInfo.get("fullscr", True))

    # Reuse the experiment’s window/monitor logic (Retina handling, sizePix logic, etc.)
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

    # Draw once immediately so something is visible even if key focus is delayed
    line.draw()
    msg.draw()
    exp.win.flip()

    # Wait for a decision key (more robust than polling loops)
    key = event.waitKeys(keyList=["c", "r", "q", "escape"])[0]

    if key in ("q", "escape"):
        exp.win.close()
        core.quit()

    if key == "c":
        print("Geometry confirmed.")
    elif key == "r":
        print("Geometry mismatch reported.")

    exp.win.close()
    core.quit()


if __name__ == "__main__":
    run_calibration(theta_deg=10.0, fullscr=True)
