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

def make_model(ω_c, T_c):
    (p_H, p_L) = ot.timings(.1, .1)

    return OttoEngine(
        δ=[.8, .8],
        ω_c=[ω_c, ω_c],
        ψ_0=qt.basis([2], [1]),
        description=f"Classic Cycle",
        k_max=5,
        bcf_terms=[6] * 2,
        truncation_scheme="simplex",
        driving_process_tolerances=[StocProcTolerances(1e-3, 1e-3)] * 2,
        thermal_process_tolerances=[StocProcTolerances(1e-3, 1e-3)] * 2,
        T=[T_c, 4],
        therm_methods=["tanhsinh", "tanhsinh"],
        Δ=1,
        num_cycles=3,
        Θ=50,
        dt=0.001,
        timings_H=p_H,
        timings_L=p_L,
        streaming_mode=True,
        shift_to_resonance=(False, False),
        L_shift=(0, 0),
    )

ωs = [round(ω, 3) for ω in np.linspace(.5, 3, 5)]
Ts = [round(T, 3) for T in np.linspace(.4, 1.5, 5)]
ωs, Ts

import itertools
models = [make_model(ω, T) for ω, T, in itertools.product(ωs, Ts)]

ot.integrate_online_multi(models, 100_000, increment=10_000, analyze_kwargs=dict(every=10_000))
