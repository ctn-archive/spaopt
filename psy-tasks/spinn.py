import nengo
import nengo.spa
from nengo.utils.numpy import rmse
import nengo_spinnaker
import numpy as np

from benchmarks import get_seed, Benchmark, SignalGenerator
import spaopt
import spaopt.optimization

nengo.log(debug=True)


class SpinnRepr(Benchmark):
    def run_param_set(self, n_neurons, d, seed, trial):
        spaopt.optimization.SubvectorRadiusOptimizer.Simulator = \
            nengo_spinnaker.Simulator

        rng = np.random.RandomState(seed)

        ctx = nengo.spa.SemanticPointer(d, rng)
        ctx.make_unitary()

        model = nengo.Network(seed=get_seed(rng))
        with model:
            step = SignalGenerator(self.duration).make_step(0, d, .001, rng)
            in_a = nengo.Node(step, size_out=d)

            a = nengo.Node(size_in=d)
            b = nengo.Node(size_in=d)
            c = nengo.Node(size_in=d)
            nengo.Connection(in_a, a)
            nengo.Connection(a, b, synapse=None)
            nengo.Connection(b, c, synapse=None)

            old_repr = nengo.networks.EnsembleArray(n_neurons, d)
            nengo.Connection(in_a, old_repr.input)

            repr_ = spaopt.UnitEA(n_neurons, d, d)
            nengo.Connection(in_a, repr_.input)

            in_probe = nengo.Probe(c, synapse=0.005)
            old_probe = nengo.Probe(old_repr.output, synapse=0.005)
            probe = nengo.Probe(repr_.output, synapse=0.005)

        nengo_spinnaker.add_spinnaker_params(model.config)
        model.config[in_a].function_of_time = True
        sim = nengo_spinnaker.Simulator(model)
        sim.run(self.duration)
        sim.close()

        return {
            't': sim.trange(),
            'default': rmse(sim.data[old_probe], sim.data[in_probe], axis=1),
            'optimized': rmse(sim.data[probe], sim.data[in_probe], axis=1)
        }
