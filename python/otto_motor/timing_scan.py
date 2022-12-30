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

def timings(τ_c, τ_i, percent_overlap=0):
    τ_cI = τ_c * (1-percent_overlap)

    τ_thI = (1 - 2 * τ_cI) / 2
    τ_th = (1 - 2 * τ_c) / 2
    τ_i_on = (τ_thI - 2*τ_i)
    timings_H = (0, τ_c, τ_c + τ_th, 2*τ_c + τ_th)

    timings_L_hot = (τ_cI, τ_cI + τ_i, τ_cI + τ_i + τ_i_on, τ_cI + 2 * τ_i + τ_i_on)

    timings_L_cold = tuple(time + timings_H[2] for time in timings_L_hot)

    return timings_H, (timings_L_cold, timings_L_hot)

τ_mod, τ_I = 0.1, 0.1
(p_H, p_L) = timings(τ_mod, τ_I, .5)
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
    T=[1, 4],
    therm_methods=["tanhsinh", "tanhsinh"],
    Δ=1,
    num_cycles=4,
    Θ=1.5 / 0.05,
    dt=0.001,
    timings_H=p_H,
    timings_L=p_L,
    streaming_mode=True,
    shift_to_resonance=(False, False),
    L_shift=(0.0, 0.0),
)
ot.plot_cycle(prototype)

overlaps = np.round(np.linspace(0, 1, 3), 3)
shifts = np.round(np.linspace(0, τ_mod, 3), 3)

models = []

import itertools

for overlap, shift in itertools.product(overlaps, shifts):
    print(overlap, shift)
    (p_H, p_L) = timings(τ_mod, τ_I, overlap)

    model = prototype.copy()
    model.timings_H = p_H
    model.timings_L = p_L
    model.L_shift = (shift, shift)
    models.append(model)


ot.plot_cycles(models)
