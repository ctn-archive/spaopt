import os.path
import sys

sys.path.insert(0, os.path.dirname(__file__))

import platform

import numpy as np
from psyrun import Param
from psyrun.scheduler import Sqsub

from benchmarks import get_seed, EmpiricalError


rng = np.random.RandomState(29873)
n_trials = 20
seeds = [get_seed(rng) for i in range(n_trials)]


pspace = (Param(dimensions=[64, 256]) * Param(N=[50, 200]) *
          Param(r=np.linspace(0.01, 1, 11)) *
          Param(seed=seeds, trial=range(n_trials)))
min_items = 64


if platform.node().startswith('bul') or platform.node().startswith('saw'):
    workdir = '/work/jgosmann/spaopt'
    scheduler = Sqsub(workdir)
    scheduler_args = {
        'timelimit': '15m',
        'memory': '1024M'
    }
else:
    workdir = os.path.join(os.path.dirname(__file__), os.pardir, 'psywork')


def execute(**params):
    return EmpiricalError().run_param_set(**params)
