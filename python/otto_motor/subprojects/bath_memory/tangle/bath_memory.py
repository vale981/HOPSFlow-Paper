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

def make_shift_model(shift_c, shift_h, switch_t=3):
    switch_time = switch_t / 50

    (p_H, p_L) = ot.timings(switch_time, switch_time)
    return OttoEngine(
        δ=[.7, .7],
        ω_c=[1, 1],
        ψ_0=qt.basis([2], [1]),
        description=f"Classic Cycle",
        k_max=5,
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

def overlap(shift_model, N, step, switch_t=6):
    switch_time = switch_t / 50
    next_model = shift_model.copy()
    (p_H, p_L) = ot.timings(switch_time, switch_time)

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

def make_model(ω_c, T_c):
    best_shift_model = make_shift_model(.12, .12)
    new_step_size = 6
    mini_step = .12


    overlapped_model = overlap(best_shift_model, 1, mini_step, new_step_size)
    overlapped_model.T[0] = T_c
    overlapped_model.ω_c = [ω_c, ω_c]
    return overlapped_model

ωs = [round(ω, 3) for ω in np.linspace(.5, 1.5, 5)]
Ts = [round(T, 3) for T in np.linspace(.4, .6, 5)]
ωs, Ts

import itertools
models = [make_model(ω, T) for ω, T, in itertools.product(ωs, Ts)]

ot.integrate_online_multi(models, 30_000, increment=10_000, analyze_kwargs=dict(every=10_000))

models[1].T

fig, ax = plt.subplots()
for model in models[:22]:
    pu.plot_with_σ(models[0].t, model.interaction_power().sum_baths().integrate(model.t), ax=ax)
    print(model.power(steady_idx=2).value, model.T[0], model.ω_c[0])

fig, ax = plt.subplots()
for model in models[:22]:
  pu.plot_with_σ(models[0].t, model.system_energy(), ax=ax)

pu.plot_with_σ(models[0].t, models[0].interaction_power().sum_baths().integrate(models[0].t))

ot.plot_power_eff_convergence(models[:10], 2)

f = plt.figure()
a_power = f.add_subplot(121, projection="3d")
a_efficiency = f.add_subplot(122, projection="3d")
for ax in [a_power, a_efficiency]:
    ax.set_box_aspect(aspect=None, zoom=0.85)
    ax.set_xlabel(r"$T_c$")
    ax.set_ylabel(r"$\omega_c$")

ot.plot_3d_heatmap(
    models[:20],
    lambda model: np.clip(-model.power(steady_idx=2).value, 0, np.inf),
    lambda model: model.T[0],
    lambda model: model.ω_c[0],
    ax=a_power,
)
a_power.set_zlabel(r"$P$")


ot.plot_3d_heatmap(
    models[:20],
    lambda model: np.clip(np.nan_to_num(model.efficiency(steady_idx=2).value * 100), 0, np.inf),
    lambda model: model.T[0],
    lambda model: model.ω_c[0],
    ax=a_efficiency,
)
a_efficiency.set_zlabel(r"$\eta$")
fs.export_fig("bath_memory_power_efficiency")
