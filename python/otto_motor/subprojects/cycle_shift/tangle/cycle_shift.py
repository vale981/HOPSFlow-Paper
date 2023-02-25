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

T = 50
switch_time = 3. / T
def make_model(shift_c, shift_h):
    (p_H, p_L) = ot.timings(switch_time, switch_time)
    return OttoEngine(
        δ=[.7, .7],
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
        L_shift=(shift_c, shift_h),
    )

N = 3
N_over = 1
step = switch_time / (N-N_over)
shifts = [round(shift * step, 3) for shift in range(-N, N+1)]
shifts

import itertools
models = [make_model(shift, shift) for shift in shifts]

ot.integrate_online_multi(models, 100_000, increment=10_000, analyze_kwargs=dict(every=10_000))
