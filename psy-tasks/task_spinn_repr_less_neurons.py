import os.path
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from psyrun import Param

from benchmarks import get_seed, LessNeuronParams
from spinn import SpinnRepr

rng = np.random.RandomState(128)
n_trials = 20
seeds = [get_seed(rng) for i in range(n_trials)]


workdir = os.path.join(os.path.dirname(__file__), os.pardir, 'psywork')
try:
    pspace = LessNeuronParams(os.path.join(workdir, 'spinn_repr', 'result.h5'))
except IOError:
    pspace = Param()


def execute(**params):
    params = dict(params)
    params['n_neurons'] = params['reduced_neurons']
    del params['reduced_neurons']
    return SpinnRepr().run_param_set(**params)
