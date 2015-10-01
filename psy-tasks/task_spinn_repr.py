import os.path
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from psyrun import Param

from benchmarks import get_seed
from spinn import SpinnRepr

rng = np.random.RandomState(128)
n_trials = 20
seeds = [get_seed(rng) for i in range(n_trials)]

pspace = (
    Param(d=25, n_neurons=200) * Param(seed=seeds, trial=range(n_trials)))


def execute(**params):
    return SpinnRepr().run_param_set(**params)
