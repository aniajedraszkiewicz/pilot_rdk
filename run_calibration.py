# run_calibration.py
# Place in pilot_rdk/ (project root, same level as experiment_files/)
# Open and run THIS file from PsychoPy.

import os
os.environ["RDK_KEYBOARD_BACKEND"] = "ptb"  # use ptb for PsychoPy standalone

from experiment_files.calibration import run_calibration

run_calibration(theta_deg=10.0)