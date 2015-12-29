import nengo
import numpy as np
from psyrun import Param
from psyrun.scheduler import Sqsub
from scipy.special import beta

def execute(d, scaling, encoders, seed):
    scalings = {
        'linear': d,
        'sqrtbeta': d * beta((d - 1) / 2., .5) / beta((d - 1) / 2., 1.),
        'quadratic': d * d,
    }
    n = int(20 * scalings[scaling])
    if n < d:
        return {
            'n': n,
            'e_noise': np.nan,
        }

    encoder_params = {
        'uniform': {
            'n_neurons': n,
            'n_ensembles': 1,
            'ens_dimensions': d,
        },
        'orthonormal': {
            'n_neurons': n // d,
            'n_ensembles': d,
            'ens_dimensions': 1,
        },
        'scaled orthonormal': {
            'n_neurons': n // d,
            'n_ensembles': d,
            'ens_dimensions': 1,
            'radius': 3.5 / np.sqrt(d),
        },
    }

    model = nengo.Network(seed=seed)
    with model:
        stim = nengo.Node(np.zeros(d))
        ea = nengo.networks.EnsembleArray(**encoder_params[encoders])
        nengo.Connection(stim, ea.input)
        p = nengo.Probe(ea.output, synapse=0.005)

    sim = nengo.Simulator(model)
    sim.run(1.)

    data = sim.data[p][200:]
    return {
        'n': n,
        'e_noise': np.mean(np.linalg.norm(
            data - np.mean(data, axis=0), axis=1)),
    }


pspace = (
    Param(scaling=['linear', 'sqrtbeta', 'quadratic']) * (
        Param(d=[4, 8, 12, 16]) *
        Param(encoders=['uniform', 'orthonormal', 'scaled orthonormal']) +
        Param(d=[24, 32, 64, 128, 256]) *
        Param(encoders=['orthonormal', 'scaled orthonormal'])) *
    Param(seed=range(20)))
