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
def make_model(shift_c, shift_h, switch_t=3.):
    switch_time = switch_t / T

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
N_over = 2
step = 3. / (T*(N-N_over))
shifts = [round(shift * step, 3) for shift in range(-N, N+1)]
shifts

import itertools
models = [make_model(shift, shift) for shift in shifts]

for model in models:
  print(model.power(steady_idx=1).value / models[3].power(steady_idx=1).value, model.efficiency(steady_idx=1).value * 100)

ot.plot_power_eff_convergence(models)

ot.plot_powers_and_efficiencies(shifts, models)

best_shift = shifts[-2]
best_shift_model = make_model(best_shift, best_shift)

def overlap(shift_model, N, step, switch_t=3.):
    switch_time = switch_t / T
    (p_H, p_L) = ot.timings(switch_time, switch_time)
    next_model = shift_model.copy()

    #next_model.timings_H=p_H
    next_model.timings_L=p_L

    (a, b, c, d) = next_model.timings_L[0]
    (e, f, g, h) = next_model.timings_L[1]
    next_step = step * N
    (s1, s2) = next_model.L_shift
    next_model.L_shift = (s1 + next_step, s2 - next_step)
    next_model.timings_L = (
        (a - 2 * next_step, b - 2 * next_step, c, d),
        (e, f, g + 2 * next_step, h + 2 * next_step),
    )
    return next_model


def overlap_cold(shift_model, N, step):
    next_model = shift_model.copy()
    (a, b, c, d) = next_model.timings_L[0]
    (e, f, g, h) = next_model.timings_L[1]
    next_step = step * N
    (s1, s2) = next_model.L_shift
    next_model.L_shift = (s1 + next_step, s2 - next_step)
    next_model.timings_L = (
        (a - 2 * next_step, b - 2 * next_step, c - next_step, d - next_step),
        (e + next_step, f + next_step, g + 2 * next_step, h + 2 * next_step),
    )
    return next_model


Ns = list(range(1, 4))
overlap_models = [overlap(best_shift_model, N, step) for N in Ns]
overlap_models = [overlap_cold(best_shift_model, N, step) for N in Ns]
new_step_size = 6
mini_step = (new_step_size / (N-N_over) / T)
overlap_models = [overlap(best_shift_model, N, mini_step, new_step_size) for N in Ns]

ot.integrate_online_multi(overlap_models, 50_000, increment=10_000, analyze_kwargs=dict(every=10_000))

all_overlap_models = [best_shift_model, *overlap_models]

ot.plot_power_eff_convergence(all_overlap_models, 1)

[model.efficiency(steady_idx=2).value / best_shift_model.efficiency(steady_idx=2).value for model in all_overlap_models]

ot.plot_powers_and_efficiencies(Ns, overlap_models)

f, a = plt.subplots()
a.axhline(0)
for model in all_overlap_models:
    pu.plot_with_σ(model.t, model.interaction_power().sum_baths().integrate(model.t), ax=a)
