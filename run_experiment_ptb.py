import os
os.environ["RDK_KEYBOARD_BACKEND"] = "ptb"

from experiment_files.experiment import Experiment

if __name__ == "__main__":
    exp = Experiment()
    exp.setup()
    exp.run_experiment()
    exp.close_exp()