import figsaver as fs
import plot_utils as pu
from hiro_models.one_qubit_model import StocProcTolerances
from hiro_models.otto_cycle import OttoEngine
import hiro_models.model_auxiliary as aux
import numpy as np
import qutip as qt
import utilities as ut
import stocproc
import matplotlib.pyplot as plt
import otto_utilities as ot

import ray
ray.shutdown()

#ray.init(address='auto')
ray.init()
from hops.util.logging_setup import logging_setup
import logging
logging_setup(logging.INFO)
plt.rcParams['figure.figsize'] = (12,4)

def timings(τ_c, τ_i):
    τ_th = (1 - 2 * τ_c) / 2
    τ_i_on = τ_th - 2 * τ_i
    timings_H = (0, τ_c, τ_c + τ_th, 2 * τ_c + τ_th)
    timings_L_hot = (τ_c, τ_c + τ_i, τ_c + τ_i + τ_i_on, τ_c + 2 * τ_i + τ_i_on)

    timings_L_cold = tuple(time + timings_H[2] for time in timings_L_hot)

    return timings_H, (timings_L_cold, timings_L_hot)

def make_cycle(θ):
    (p_H, p_L) = timings(3. / θ, 3. / θ)

    return OttoEngine(
        δ=[0.4, 0.4],
        ω_c=[2, 2],
        ψ_0=qt.basis([2], [1]),
        description=f"Classic Cycle",
        k_max=3,
        bcf_terms=[4] * 2,
        truncation_scheme="simplex",
        driving_process_tolerances=[StocProcTolerances(1e-3, 1e-3)] * 2,
        thermal_process_tolerances=[StocProcTolerances(1e-3, 1e-3)] * 2,
        T=[0.5, 4],
        therm_methods=["tanhsinh", "tanhsinh"],
        Δ=1,
        num_cycles=2,
        Θ=θ,
        dt=0.001,
        timings_H=p_H,
        timings_L=p_L,
        streaming_mode=True,
        shift_to_resonance=(False, False),
        L_shift=(0, 0),
    )

long_cycle = make_cycle(70)

ot.integrate_online(long_cycle, 10000)

f, a, *_ = pu.plot_with_σ(long_cycle.t, long_cycle.system_energy())
a.set_xlim(0, long_cycle.Θ)

ot.plot_energy(long_cycle)

def thermal_state(Ω, T):
    ρ = np.array([[np.exp(-Ω/T), 0], [0, 1]])
    ρ /= np.sum(np.diag(ρ))

    return ρ

import hops.util.utilities
from hopsflow.util import EnsembleValue
with aux.get_data(long_cycle) as data:
    trace_dist_c = hops.util.utilities.trace_distance(data, relative_to=thermal_state(long_cycle.T[0], long_cycle.energy_gaps[0]))
    trace_dist_h = hops.util.utilities.trace_distance(data, relative_to=thermal_state(long_cycle.T[1], long_cycle.energy_gaps[1]))

f, a = plt.subplots()
pu.plot_with_σ(long_cycle.t, EnsembleValue(trace_dist_c), ax=a)
pu.plot_with_σ(long_cycle.t, EnsembleValue(trace_dist_h), ax=a)
a.plot(long_cycle.t, long_cycle.H(long_cycle.t)[:, 0, 0] - 1)
#a.set_xlim(155)
