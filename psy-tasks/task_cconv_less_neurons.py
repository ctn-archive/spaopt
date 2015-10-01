import os.path
import sys

sys.path.insert(0, os.path.dirname(__file__))

import platform

import numpy as np
from psyrun import Param
from psyrun.scheduler import Sqsub

from benchmarks import get_seed, CConv, LessNeuronParams


rng = np.random.RandomState(128)
n_trials = 20
seeds = [get_seed(rng) for i in range(n_trials)]
min_items = 2


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
    pspace = LessNeuronParams(os.path.join(workdir, 'cconv', 'result.h5'))
except IOError:
    pspace = Param()


def execute(**params):
    pcopy = dict(params)
    pcopy['n_neurons'] = pcopy['reduced_neurons']
    del pcopy['reduced_neurons']
    return CConv().run_param_set(**pcopy)
