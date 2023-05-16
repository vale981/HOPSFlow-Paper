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
import hops
from hopsflow.util import EnsembleValue
import ray
ray.shutdown()

#ray.init(address='auto')
ray.init()
from hops.util.logging_setup import logging_setup
import logging
logging_setup(logging.INFO)

T = 50

def make_model_orig(shift_c, shift_h, switch_t=3.0, switch_t_sys=None, only_cold=False):
    switch_time = switch_t / T
    switch_time_sys = (switch_t_sys if switch_t_sys else switch_t) / T

    (p_H, p_L) = ot.timings(switch_time_sys, switch_time)
    return OttoEngine(
        δ=[0.7, 0.7],
        ω_c=[1, 1],
        ψ_0=qt.basis([2], [1]),
        description=f"Classic Cycle",
        k_max=4,
        bcf_terms=[5] * 2,
        truncation_scheme="simplex",
        driving_process_tolerances=[StocProcTolerances(1e-3, 1e-3)] * 2,
        thermal_process_tolerances=[StocProcTolerances(1e-3, 1e-3)] * 2,
        T=[0.5, 4],
        therm_methods=["tanhsinh", "tanhsinh"],
        Δ=1,
        num_cycles=3,
        Θ=60,
        dt=0.001,
        timings_H=p_H,
        timings_L=p_L,
        streaming_mode=True,
        shift_to_resonance=(False, False),
        L_shift=(shift_c, 0 if only_cold else shift_h),
    )

def make_model(ω_c, T_c):
    model =  make_model_orig(0, 0, switch_t = 6.)


    model.T[0] = T_c
    model.ω_c = [ω_c, ω_c]
    return model

ωs = [round(ω, 3) for ω in np.linspace(.5, 1.5, 5)]
Ts = [round(T, 3) for T in np.linspace(.4, .6, 5)]
ωs, Ts

import itertools
models = [make_model(ω, T) for ω, T, in itertools.product(ωs, Ts)]

ot.integrate_online_multi(models, 30_000, increment=10_000, analyze_kwargs=dict(every=10_000))
