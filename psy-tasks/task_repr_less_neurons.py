import os.path
import sys

sys.path.insert(0, os.path.dirname(__file__))

import platform

import numpy as np
from psyrun import Param
from psyrun.scheduler import Sqsub

from benchmarks import get_seed, Repr, LessNeuronParams

rng = np.random.RandomState(128)
n_trials = 20
seeds = [get_seed(rng) for i in range(n_trials)]


if platform.node().startswith('bul') or platform.node().startswith('saw'):
    workdir = '/work/jgosmann/spaopt'
    scheduler = Sqsub(workdir)
    scheduler_args = {
        'timelimit': '240m',
        'memory': '2048M'
    }
else:
    workdir = os.path.join(os.path.dirname(__file__), os.pardir, 'psywork')

try:
    pspace = LessNeuronParams(os.path.join(workdir, 'repr', 'result.h5'))
except IOError:
    pspace = Param()


def execute(**params):
    params = dict(params)
    params['n_neurons'] = params['reduced_neurons']
    del params['reduced_neurons']
    return Repr().run_param_set(**params)
