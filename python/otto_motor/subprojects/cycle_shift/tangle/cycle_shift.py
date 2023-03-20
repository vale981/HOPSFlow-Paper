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
    print(switch_time * 60)
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
extra_r = 2
step = 3. / (T*(N-N_over))
shifts = [round(shift * step, 3) for shift in range(-N, N+1+extra_r)]
shifts

import itertools
models = [make_model(shift, shift) for shift in shifts]
baseline = models[3]

ot.plot_cycle(baseline)
fs.export_fig("cycle_prototype", y_scaling=.7)

ot.integrate_online_multi(models, 80_000, increment=10_000, analyze_kwargs=dict(every=10_000))

for model in models:
  print(model.power(steady_idx=1).value / baseline.power(steady_idx=1).value, model.efficiency(steady_idx=1).value)

ot.plot_energy(baseline)
print(
    fs.tex_value(baseline.system_energy().N,  prefix="N="),
  )

fs.export_fig("prototype_full_energy", x_scaling=2, y_scaling=1)

def thermal_state(T, Ω):
    ρ = np.array([[np.exp(-Ω/T), 0], [0, 1]])
    ρ /= np.sum(np.diag(ρ))

    return ρ
import hops.util.utilities
from hopsflow.util import EnsembleValue
for model in models[3:4]:
    with aux.get_data(model) as data:
        trace_dist_c = hops.util.utilities.trace_distance(data, relative_to=thermal_state(model.T[0], model.energy_gaps[0]))
        trace_dist_h = hops.util.utilities.trace_distance(data, relative_to=thermal_state(model.T[1], model.energy_gaps[1]))
        f, (a, aa) = plt.subplots(nrows=1, ncols=2)
        print(thermal_state(model.T[0], model.energy_gaps[0]))
        print(thermal_state(model.T[1], model.energy_gaps[1]))
        pu.plot_with_σ(model.t, EnsembleValue(trace_dist_c), ax=a, label=r"$||\rho(\tau)-\rho_c||$")
        pu.plot_with_σ(model.t, EnsembleValue(trace_dist_h), ax=a, label=r"$||\rho(\tau)-\rho_h||$")
        aa.plot(model.t, data.rho_t_accum.mean[:,0,0].real,  label=r"$\rho_{00}$")
        aa.axhline(thermal_state(model.T[1], model.energy_gaps[1])[0,0],  label=r"$\rho_{h,00}$", color="lightgray")
        aa.axhline(thermal_state(model.T[0], model.energy_gaps[0])[0,0],  label=r"$\rho_{c,00}$", color="lightgray")


        a.set_xlim(2*model.Θ, 3*model.Θ)
        aa.set_xlim(2*model.Θ, 3*model.Θ)
        a.plot(model.t, (model.H(model.t)[:, 0, 0] - 1)/2, label="$H_\mathrm{sys}$ Modulation")
        a.set_xlabel(r"$\tau$")
        aa.set_xlabel(r"$\tau$")
        #a.set_xlim(155)
        a.legend()
        aa.legend()
        aa.set_ylim((0.1,.4))
        fs.export_fig("prototype_thermalization", y_scaling=.7, x_scaling=2)

ot.plot_steady_energy_changes([baseline], 2, label_fn=lambda _: "")
fs.export_fig("prototype_energy_change", y_scaling=.7)

ot.plot_power_eff_convergence(models)
fs.export_fig("cycle_shift_convergence", x_scaling=2, y_scaling=.7)

ot.plot_powers_and_efficiencies(np.array(shifts) * 100, models, xlabel="Cycle Shift")
fs.export_fig("cycle_shift_power_efficiency", y_scaling=.7, x_scaling=1)

fig, ax =ot.plot_steady_energy_changes([baseline, models[3+2]], 2, label_fn=lambda m: ("baseline" if m.hexhash == baseline.hexhash else "shifted"))
ax.legend(loc="lower left")
fs.export_fig("shift_energy_change", y_scaling=.7)

best_shift = shifts[3+2]#[np.argmax([-model.power(steady_idx=2).value for model in models])]
best_shift_model = make_model(best_shift, best_shift)

t_shift_begin = (2 - best_shift) * baseline.Θ
t_begin = 2 * baseline.Θ
t_shift_end = (3 - best_shift) * baseline.Θ
final_period_idx = np.argmin(abs(baseline.t - t_begin))
final_period_shifted = np.argmin(abs(baseline.t - t_shift_begin))
final_period_shifted_end = final_period_shifted - final_period_idx

t_baseline = baseline.t[final_period_shifted:final_period_shifted_end]
t_final_period = baseline.t[final_period_idx:]
t_plot = baseline.t[: len(t_baseline)]
interaction_change_baseline_cold = (
    baseline.interaction_power()
    .for_bath(0)
    .slice(slice(final_period_shifted, final_period_shifted_end))
    .value
)
interaction_change_best_cold = (
    best_shift_model.interaction_power()
    .for_bath(0)
    .slice(slice(final_period_idx, len(baseline.t)))
    .value
)
interaction_change_baseline_hot = (
    baseline.interaction_power()
    .for_bath(1)
    .slice(slice(final_period_shifted, final_period_shifted_end))
    .value
)
interaction_change_best_hot = (
    best_shift_model.interaction_power()
    .for_bath(1)
    .slice(slice(final_period_idx, len(baseline.t)))
    .value
)


fig, ax = plt.subplots()
ax.plot(t_plot, interaction_change_baseline_cold, label="baseline")
ax.plot(t_plot, interaction_change_best_cold, label="shifted")
ax.plot(t_plot, interaction_change_baseline_hot, linestyle="--", color="C0")
ax.plot(t_plot, interaction_change_best_hot, linestyle="--", color="C1")
ax.legend()
ax.set_xlabel(r"$\tau$")
ax.set_ylabel(r"$P_{\mathrm{int}}$")
fs.export_fig("shift_power", y_scaling=0.7)

ot.plot_energy(baseline)
f, a = ot.plot_energy(best_shift_model)
a.plot(best_shift_model.t, best_shift_model.H(best_shift_model.t)[:, 0,0])

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


Ns = list(range(1, 4))[:1]
overlap_models = [overlap(best_shift_model, N, step) for N in Ns]
overlap_models = [overlap_cold(best_shift_model, N, step) for N in Ns]
new_step_size = 6
mini_step = (new_step_size / (N-N_over) / T)
print(mini_step)
overlap_models = [overlap(best_shift_model, N, mini_step, new_step_size) for N in Ns]

ot.integrate_online_multi(overlap_models, 80_000, increment=10_000, analyze_kwargs=dict(every=10_000))

all_overlap_models = [best_shift_model, *overlap_models]

ot.plot_power_eff_convergence(all_overlap_models, 1)

[model.power(steady_idx=2).value / best_shift_model.power(steady_idx=2).value for model in all_overlap_models]

[model.efficiency(steady_idx=2).value / best_shift_model.efficiency(steady_idx=2).value for model in all_overlap_models]

[model.power(steady_idx=2).N  for model in all_overlap_models]

ot.plot_powers_and_efficiencies([0] + Ns, all_overlap_models)

f, a = plt.subplots()
a.axhline(0, color="lightgrey")
for model, label in zip(all_overlap_models[:2], ["Shifted", "Shifted with Overlap"]):
    _, _, lines = pu.plot_with_σ(model.t, model.interaction_power().sum_baths().integrate(model.t), ax=a, label=fr"$W_\mathrm{{int}}$ {label}")
    pu.plot_with_σ(model.t, model.system_power().integrate(model.t), ax=a, color=lines[0][0].get_color(), linestyle="--", label=fr"$W_\mathrm{{sys}}$ {label}")
a.set_ylabel(r"$W_{\mathrm{int/sys}}$")
a.set_xlabel(r"$\tau$")
a.legend()
fs.export_fig("cycle_shift_shift_vs_overlap_power", x_scaling=2, y_scaling=.6)

fig, ax =ot.plot_steady_energy_changes(all_overlap_models, 2, label_fn=(lambda m: ["without overlap", "with overlap"][all_overlap_models.index(m)]))
ax.legend(loc="lower left")
fs.export_fig("overlap_energy_change", y_scaling=.9)
