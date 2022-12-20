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
    τ_i_on = (τ_th - 2*τ_i)
    timings_H = (0, τ_c, τ_c + τ_th, 2*τ_c + τ_th)
    timings_L_hot = (τ_c, τ_c + τ_i, τ_c + τ_i + τ_i_on, τ_c + 2 * τ_i + τ_i_on)

    timings_L_cold = tuple(time + timings_H[2] for time in timings_L_hot)

    return timings_H, (timings_L_cold, timings_L_hot)

(p_H, p_L) = timings(0.1, 0.3)
prototype = OttoEngine(
    δ=[0.4, 0.4],
    ω_c=[2, 2],
    ψ_0=qt.basis([2], [1]),
    description=f"A model for scanning coupling strength and interactin switch times.",
    k_max=4,
    bcf_terms=[6] * 2,
    truncation_scheme="simplex",
    driving_process_tolerances=[StocProcTolerances(1e-3, 1e-3)] * 2,
    thermal_process_tolerances=[StocProcTolerances(1e-3, 1e-3)] * 2,
    # driving_process_tolerances=[StocProcTolerances(1e-5, 1e-5)] * 2,
    # thermal_process_tolerances=[StocProcTolerances(1e-5, 1e-5)] * 2,
    T=[1, 4],
    therm_methods=["tanhsinh", "tanhsinh"],
    Δ=1,
    num_cycles=5,
    Θ=1.5 / 0.05,
    dt=0.01/8,
    timings_H=p_H,
    timings_L=p_L,
    streaming_mode=True,
    shift_to_resonance=(False, False),
)

δs = np.round(np.linspace(.3, .5, 3), 3)
τ_Is = np.array([# .05,
                 .1, .15, .2])
δs, τ_Is

models = []

import itertools

for τ_I, δ in itertools.product(τ_Is, δs):
    (p_H, p_L) = timings(0.1, τ_I)

    model = prototype.copy()
    model.δ = [δ, δ]
    model.timings_H = p_H
    model.timings_L = p_L
    models.append(model)


ot.plot_cycles(models[:: len(δs)])
