import os.path
import sys

sys.path.insert(0, os.path.dirname(__file__))

import platform

import numpy as np
from psyrun import Param
from psyrun.scheduler import Sqsub

from benchmarks import get_seed, CConv


rng = np.random.RandomState(128)
n_trials = 20
seeds = [get_seed(rng) for i in range(n_trials)]


pspace = (Param(d=[64, 64, 256], n_neurons=[50, 200, 200]) *
          Param(seed=seeds, trial=range(n_trials)))
min_items = 5


if platform.node().startswith('bul') or platform.node().startswith('saw'):
    workdir = '/work/jgosmann/spaopt'
    scheduler = Sqsub(workdir)
    scheduler_args = {
        'timelimit': '480m',
        'memory': '2048M'
    }
else:
    workdir = os.path.join(os.path.dirname(__file__), os.pardir, 'psywork')


def execute(**params):
    return CConv().run_param_set(**params)
