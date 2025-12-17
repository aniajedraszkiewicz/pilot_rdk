# main.py  (in experiment_files/)

from .experiment import Experiment


if __name__ == "__main__":
    exp = Experiment()
    exp.setup()
    exp.run_experiment()
    exp.close_exp()
