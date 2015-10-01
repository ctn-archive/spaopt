import functools


import nengo
import nengo.spa
import spaopt
from nengo.utils.numpy import rmse
from nengo.processes import Process, WhiteSignal
import numpy as np
import pandas as pd
from psyrun.io import load_dict_h5 as load_results
from psyrun.pspace import _PSpaceObj


def get_seed(rng):
    return rng.randint(np.iinfo(np.int32).max)


class LessNeuronParams(_PSpaceObj):
    def __init__(self, filename, square=True):
        data = load_results(filename)
        n_tiles = max(np.atleast_2d(x).shape[1] for x in data.values())
        data = {
            k: np.tile(v, (n_tiles, 1)).T.flat
               if len(v.shape) == 1 else v.flat for k, v in data.items()}
        super(LessNeuronParams, self).__init__(
            ['d', 'reduced_neurons', 'n_neurons', 'seed', 'trial'])

        data = pd.DataFrame({k: v for k, v in data.items()})
        mean = data.query('t > 0.5').groupby(['n_neurons', 'd', 'seed']).mean()
        mean.reset_index(inplace=True)
        self._len = len(mean)
        grand_mean = data.query('t > 0.5').groupby(
            ['n_neurons', 'd']).mean().reset_index()
        ratio = grand_mean['optimized'] / grand_mean['default']
        if square:
            ratio = np.square(ratio)
        grand_mean['reduced_neurons'] = np.maximum(5,
            (grand_mean['n_neurons'] * ratio).astype(int))
        grand_mean.drop(
            ['seed', 'default', 'optimized', 't', 'trial'], axis=1,
            inplace=True)
        self._params = pd.merge(mean, grand_mean, on=['n_neurons', 'd'])
        self._params['trial'] = self._params['trial'].astype(int)
        self._params['n_neurons'] = self._params['n_neurons'].astype(int)
        self._params['d'] = self._params['d'].astype(int)
        self._params['seed'] = self._params['seed'].astype(int)

    def iterate(self):
        for i in range(len(self)):
            yield {k: self._params.iloc[i][k] for k in self.keys()}

    def __len__(self):
        return self._len


class SignalGenerator(Process):
    def __init__(self, duration, high=5.):
        self._whitenoise = WhiteSignal(duration, high=high)

    def make_step(self, size_in, size_out, dt, rng=np.random):
        return functools.partial(
            self.sample,
            sample_whitenoise=self._whitenoise.make_step(size_in, size_out, dt, rng))

    @staticmethod
    def sample(t, sample_whitenoise):
        sampled = sample_whitenoise(t)
        return sampled / np.linalg.norm(sampled)


class Benchmark(object):
    def __init__(self, duration=10., dt=0.001):
        self.duration = duration
        self.dt = dt

    def run_param_set(self, params):
        raise NotImplementedError()


class EmpiricalError(Benchmark):
    def run_param_set(
            self, seed, r, dimensions, N, trial, m=1, reg=0.1, limit=1.,
            **ens_kwargs):
        seed = int(seed)
        N = int(N)
        dimensions = int(dimensions)

        init_duration = 0.5

        rng = np.random.RandomState(seed)

        model = nengo.Network(seed=get_seed(rng))
        with model:
            signal_fn = SignalGenerator(self.duration)
            in_node = nengo.Node(signal_fn, size_out=dimensions)
            ensemble = nengo.Ensemble(
                N, m, radius=r, neuron_type=nengo.LIFRate(), **ens_kwargs)
            direct = nengo.Ensemble(1, m, neuron_type=nengo.Direct())
            nengo.Connection(in_node[:m], ensemble)
            nengo.Connection(in_node[:m], direct)
            probe = nengo.Probe(
                ensemble, synapse=None, solver=nengo.solvers.LstsqL2(reg=reg))
            dprobe = nengo.Probe(direct, synapse=None)

        assert self.duration > init_duration
        sim = nengo.Simulator(model, dt=self.dt)
        sim.run(self.duration, progress_bar=False)

        selection = sim.trange() > init_duration
        return {'mse': np.sum(np.mean(np.square(
            sim.data[probe][selection, :] - sim.data[dprobe][selection, :]),
            axis=0))}


class CConv(Benchmark):
    def run_param_set(self, n_neurons, d, seed, trial):
        seed = int(seed)
        n_neurons = int(n_neurons)
        d = int(d)

        rng = np.random.RandomState(seed)

        ctx = nengo.spa.SemanticPointer(d, rng)
        ctx.make_unitary()

        model = nengo.Network(seed=get_seed(rng))
        with model:
            in_a = nengo.Node(SignalGenerator(self.duration), size_out=d)
            in_b = nengo.Node(output=ctx.v)

            old_cconv = nengo.networks.CircularConvolution(n_neurons, d)
            old_result = nengo.Ensemble(1, d, neuron_type=nengo.Direct())
            nengo.Connection(in_a, old_cconv.A)
            nengo.Connection(in_b, old_cconv.B)
            nengo.Connection(old_cconv.output, old_result)

            cconv = spaopt.CircularConvolution(n_neurons, d)
            result = nengo.Ensemble(1, d, neuron_type=nengo.Direct())
            nengo.Connection(in_a, cconv.A)
            nengo.Connection(in_b, cconv.B)
            nengo.Connection(cconv.output, result)

            with nengo.Network() as net:
                net.config[nengo.Ensemble].neuron_type = nengo.Direct()
                d_cconv = nengo.networks.CircularConvolution(1, d)
                d_result = nengo.Ensemble(1, d)
            nengo.Connection(in_a, d_cconv.A)
            nengo.Connection(in_b, d_cconv.B)
            nengo.Connection(d_cconv.output, d_result)

            old_probe = nengo.Probe(old_result, synapse=None)
            probe = nengo.Probe(result, synapse=None)
            d_probe = nengo.Probe(d_result, synapse=None)

        sim = nengo.Simulator(model)
        sim.run(self.duration, progress_bar=False)

        return {
            't': sim.trange(),
            'default': rmse(sim.data[old_probe], sim.data[d_probe], axis=1),
            'optimized': rmse(sim.data[probe], sim.data[d_probe], axis=1)
        }


class Repr(Benchmark):
    def run_param_set(self, n_neurons, d, seed, trial):
        seed = int(seed)
        n_neurons = int(n_neurons)
        d = int(d)

        rng = np.random.RandomState(seed)

        model = nengo.Network(seed=get_seed(rng))
        with model:
            in_a = nengo.Node(SignalGenerator(self.duration), size_out=d)

            old_repr = nengo.networks.EnsembleArray(n_neurons, d)
            old_result = nengo.Ensemble(1, d, neuron_type=nengo.Direct())
            nengo.Connection(in_a, old_repr.input)
            nengo.Connection(old_repr.output, old_result)

            repr_ = spaopt.UnitEA(n_neurons, d, d)
            result = nengo.Ensemble(1, d, neuron_type=nengo.Direct())
            nengo.Connection(in_a, repr_.input)
            nengo.Connection(repr_.output, result)

            with nengo.Network() as net:
                net.config[nengo.Ensemble].neuron_type = nengo.Direct()
                d_repr = nengo.networks.EnsembleArray(1, d)
                d_result = nengo.Ensemble(1, d)
            nengo.Connection(in_a, d_repr.input)
            nengo.Connection(d_repr.output, d_result)

            old_probe = nengo.Probe(old_result, synapse=None)
            probe = nengo.Probe(result, synapse=None)
            d_probe = nengo.Probe(d_result, synapse=None)

        sim = nengo.Simulator(model)
        sim.run(self.duration, progress_bar=False)

        return {
            't': sim.trange(),
            'default': rmse(sim.data[old_probe], sim.data[d_probe], axis=1),
            'optimized': rmse(sim.data[probe], sim.data[d_probe], axis=1)
        }


class Dot(Benchmark):
    def run_param_set(self, n_neurons, d, seed, trial):
        seed = int(seed)
        n_neurons = int(n_neurons)
        d = int(d)

        rng = np.random.RandomState(seed)

        ctx = nengo.spa.SemanticPointer(d, rng)
        ctx.make_unitary()

        model = nengo.Network(seed=get_seed(rng))
        with model:
            in_a = nengo.Node(SignalGenerator(self.duration), size_out=d)
            in_b = nengo.Node(output=ctx.v)

            old_prod = nengo.networks.Product(n_neurons, d)
            old_result = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
            nengo.Connection(in_a, old_prod.A)
            nengo.Connection(in_b, old_prod.B)
            nengo.Connection(
                old_prod.output, old_result,
                transform=nengo.networks.product.dot_product_transform(d))

            prod = spaopt.Product(n_neurons, d)
            result = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
            nengo.Connection(in_a, prod.A)
            nengo.Connection(in_b, prod.B)
            nengo.Connection(
                prod.output, result,
                transform=nengo.networks.product.dot_product_transform(d))

            with nengo.Network() as net:
                net.config[nengo.Ensemble].neuron_type = nengo.Direct()
                d_prod = nengo.networks.EnsembleArray(2, d, 2)
                d_result = nengo.Ensemble(1, 1)
            nengo.Connection(in_a, d_prod.input[::2])
            nengo.Connection(in_b, d_prod.input[1::2])
            nengo.Connection(
                d_prod.add_output('dot', lambda x: x[0] * x[1]), d_result,
                transform=[d * [1.]])

            old_probe = nengo.Probe(old_result, synapse=None)
            probe = nengo.Probe(result, synapse=None)
            d_probe = nengo.Probe(d_result, synapse=None)

        sim = nengo.Simulator(model)
        sim.run(self.duration, progress_bar=False)

        return {
            't': sim.trange(),
            'default': rmse(sim.data[old_probe], sim.data[d_probe], axis=1),
            'optimized': rmse(sim.data[probe], sim.data[d_probe], axis=1)
        }
