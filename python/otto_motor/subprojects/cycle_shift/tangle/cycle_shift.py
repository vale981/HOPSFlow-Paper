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
import shift_cycle as sc
import ray
ray.shutdown()

#ray.init(address='auto')
ray.init()
from hops.util.logging_setup import logging_setup
import logging
logging_setup(logging.INFO)
plt.rcParams['figure.figsize'] = (12,4)

ot.plot_cycle(baseline)
fs.export_fig("cycle_prototype", y_scaling=.7)

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

ot.plot_bloch_components(baseline)
fs.export_fig("state_evolution", y_scaling=.7)

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
best_shift_model = sc.make_model(best_shift, best_shift)

ot.plot_bloch_components(best_shift_model)

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

f, a = plt.subplots()
a.axhline(best_shift_model.system_energy().value[np.argmin(abs(best_shift_model.t - model.Θ * 2))], color="gray", linestyle="--")
r = pu.plot_with_σ(
    best_shift_model.t, best_shift_model.interaction_energy().for_bath(0), ax=a,
    label=r"$\langle H_\mathrm{inter}\rangle$"
)
pu.plot_with_σ(
    best_shift_model.t, best_shift_model.system_energy(), ax=a, label=r"$\langle H_\mathrm{sys}\rangle$"
)
# a.plot(best_shift_model.t, best_shift_model.H(best_shift_model.t)[:, 0,0])
a.plot(
    best_shift_model.t,
    best_shift_model.coupling_operators[0].operator_norm(best_shift_model.t) / 5,
    label="cold bath modulation",
)

# a.plot(
#     best_shift_model.t,
#     best_shift_model.coupling_operators[1].operator_norm(best_shift_model.t) / 5,
#     label="hot bath modulation",
# )
a.plot(
    best_shift_model.t, best_shift_model.system.operator_norm(best_shift_model.t) / 5,
    label="system modulation"
)
# a.plot(best_shift_model.t, best_shift_model.coupling_operators[1].operator_norm(best_shift_model.t) / 5)
a.set_xlim((model.Θ * 2, model.Θ * 2 + 7))

a.set_ylim((-.21, .45))
a.set_xlabel(r"$\tau$")
a.legend(loc="upper right", fontsize="x-small")
fs.export_fig("cold_bath_decoupling", y_scaling=.6)

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

all_overlap_models = [best_shift_model, *overlap_models]

ot.integrate_online_multi(overlap_models, 80_000, increment=10_000, analyze_kwargs=dict(every=10_000))

ot.plot_power_eff_convergence(all_overlap_models, 2)

f, a= ot.plot_energy(all_overlap_models[-1])
a.plot(model.t, model.coupling_operators[0].operator_norm(model.t))
a.plot(model.t, model.coupling_operators[1].operator_norm(model.t))
a.plot(model.t, model.system.operator_norm(model.t))

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

fig, ax =ot.plot_steady_work_baths(all_overlap_models, 2, label_fn=(lambda m: ["without overlap", "with overlap"][all_overlap_models.index(m)]))
ax.legend(loc="lower left")

fs.export_fig("overlap_energy_change_hot_cold", y_scaling=.9)

r = pu.plot_with_σ(all_overlap_models[-1].t, all_overlap_models[-1].interaction_energy().for_bath(0))
# a.plot(all_overlap_models[-1].t, all_overlap_models[-1].H(all_overlap_models[-1].t)[:, 0,0])
r[1].plot(all_overlap_models[-1].t, all_overlap_models[-1].coupling_operators[0].operator_norm(all_overlap_models[-1].t) / 5)
r[1].plot(all_overlap_models[-1].t, all_overlap_models[-1].coupling_operators[1].operator_norm(all_overlap_models[-1].t) / 5)
r[1].set_xlim((model.Θ*2, model.Θ*2+15))

from itertools import cycle
lines = ["--","-.",":", "-"]
linecycler = cycle(lines)

fig, ax = plt.subplots()
t = np.linspace(0, long_models[0].Θ, 1000)
l, = ax.plot(t, long_models[0].H.operator_norm(t)/2-.5, linewidth=3, color="lightgrey")
legend_1 = ax.legend([l], [r"$(||H||-1)/2$"], loc="center left", title="Reference")
from cycler import cycler
for model in [best_shift_model, long_models[5]]:
    ax.plot(t, model.coupling_operators[1].operator_norm(t), label=fr"${model.L_shift[0] * 100:.0f}\%$", linestyle=(next(linecycler)))
    #ax.plot(t, model.coupling_operators[0].operator_norm(t), label=fr"${model.L_shift[0] * 100:.0f}\%$", linestyle=(next(linecycler)))
ax.legend(title=r"Shift of $L_h$", fontsize="x-small", ncols=2)
ax.set_xlabel(r"$\tau$")
ax.set_ylabel(r"Operator Norm")
ax.add_artist(legend_1)
ax.set_xlim((0, long_models[0].Θ))
fs.export_fig("cycle_shift_long_shifts", x_scaling=2, y_scaling=.5)

long_baseline = long_models[np.argmin(abs(np.array(shifts) - 0))]
long_baseline = best_shift_model
for shift, model in zip(shifts, long_models):
    print(
        shift, best_shift,
        model.power(steady_idx=2).N,
        model.power(steady_idx=2).value / long_baseline.power(steady_idx=2).value,
        (model.efficiency(steady_idx=2).value - long_baseline.efficiency(steady_idx=2).value) * 100,
        (model.efficiency(steady_idx=2).value, long_baseline.efficiency(steady_idx=2).value),
    )

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
_, ax1_right = ot.plot_powers_and_efficiencies(np.array(shifts) * 100, models, xlabel="Cycle Shift", ax=ax1)[2]
_, ax2_right = ot.plot_powers_and_efficiencies(np.array(shifts) * 100, long_models, xlabel="Cycle Shift", ax=ax2)[2]

ax1_right.sharey(ax2_right)
ax1.sharey(ax2)

ax1.set_title("Fast Coupling")
ax2.set_title("Slow Coupling")
fs.export_fig("cycle_shift_power_efficiency_with_slower", y_scaling=.7, x_scaling=2)

fig, ax =ot.plot_steady_energy_changes([long_models[3+2], models[3+2]], 2, label_fn=lambda m: ("long" if m.hexhash == long_models[3+2].hexhash else "short"))
ax.legend(loc="lower left")

#fs.export_fig("shift_energy_change", y_scaling=.7)

powers_long = [-model.power(steady_idx=2).value for model in long_models]
powers_short = [-model.power(steady_idx=2).value for model in models]
power_overlap = -overlap_models[0].power(steady_idx=2).value
plt.plot(shifts, powers_short)
plt.plot(shifts, powers_long)
plt.axhline(power_overlap)

efficiencys_long = [model.efficiency(steady_idx=2).value for model in long_models]
efficiencys_short = [model.efficiency(steady_idx=2).value for model in models]
efficiency_overlap = overlap_models[0].efficiency(steady_idx=2).value
plt.plot(shifts, efficiencys_short)
plt.plot(shifts, efficiencys_long)
plt.axhline(efficiency_overlap)

best_long_model = long_models[5]

flow_long = -1*best_long_model.bath_energy_flow().for_bath(0)
power_long = best_long_model.interaction_power().for_bath(0)

flow_short = -1*best_shift_model.bath_energy_flow().for_bath(0)
power_short = best_shift_model.interaction_power().for_bath(0)

plt.plot(best_shift_model.t, flow_short.value, label="fast coupling")
plt.plot(best_shift_model.t, flow_long.value, label="slow coupling")
plt.plot(best_shift_model.t, power_short.value, linestyle="--", color="C0")
plt.plot(best_shift_model.t, power_long.value, linestyle="--",  color="C1")
plt.xlim((2*best_long_model.Θ-5, 2*best_long_model.Θ+10))
plt.ylim((-.015,.06))
plt.legend()
plt.xlabel(r"$\tau$")
fs.export_fig("cold_bath_flow", y_scaling=.7)

t, rel_short_cold = ot.val_relative_to_steady(
    best_shift_model,
    best_shift_model.bath_energy().for_bath(0),
    2,
    1-best_shift_model.L_shift[0]
)

t, rel_short_hot = ot.val_relative_to_steady(
    best_shift_model,
    best_shift_model.bath_energy().for_bath(1),
    2,
    1-best_shift_model.L_shift[0]
)

t, rel_long_cold = ot.val_relative_to_steady(
    best_long_model,
    best_long_model.bath_energy().for_bath(0),
    2,
    (1-best_long_model.L_shift[0])
)
t, rel_long_hot = ot.val_relative_to_steady(
    best_long_model,
    best_long_model.bath_energy().for_bath(1),
    2,
    (1-best_long_model.L_shift[0])
)
# plt.plot(t, -(rel_long_cold).value, label="slow coupling")
# plt.plot(t, -(rel_long_hot).value, label="slow coupling")
# plt.plot(t, best_long_model.coupling_operators[1].operator_norm(t), label="slow coupling")

plt.plot(t, -(rel_long_cold/rel_long_hot).value, label="slow coupling")
plt.plot(t, -(rel_short_cold/rel_short_hot).value, label="fast coupling")
plt.plot(t, best_long_model.coupling_operators[0].operator_norm(t), color="C0", linestyle="dashed")
plt.plot(t, best_shift_model.coupling_operators[0].operator_norm(t), color="C1", linestyle="dashed")

plt.ylim((-.1,.75))
plt.xlim((100, 128))
plt.legend()
plt.xlabel(r"$\tau$")
plt.ylabel(r"$-\Delta \langle{H_{\mathrm{B},c}}\rangle/\Delta \langle{H_{\mathrm{B},h}}\rangle$")
fs.export_fig("hot_vs_cold_bath", y_scaling=.7)

plt.plot(best_shift_model.t, (best_shift_model.bath_energy().for_bath(0) / best_shift_model.bath_energy().for_bath(1)).value)
plt.ylim((-1, 1))

from itertools import cycle
lines = ["--","-.",":", "-"]
linecycler = cycle(lines)
fig, ax = plt.subplots()
t = np.linspace(0, models[0].Θ, 1000)
#l, = ax.plot(t, models[0].H.operator_norm(t)/2-.5, linewidth=3, color="lightgrey")
l, = ax.plot(t, cold_models[3].coupling_operators[1].operator_norm(t), linewidth=3, color="lightgrey")
legend_1 = ax.legend([l], [r"$(||H||-1)/2$"], loc="center left", title="Reference")
from cycler import cycler
for model in cold_models:
    ax.plot(t, model.coupling_operators[0].operator_norm(t), label=fr"${model.L_shift[0] * 100:.0f}\%$", linestyle=(next(linecycler)))
ax.legend(title=r"Shift of $L_h$", fontsize="x-small", ncols=2)
ax.set_xlabel(r"$\tau$")
ax.set_ylabel(r"Operator Norm")
ax.add_artist(legend_1)
ax.set_xlim((0, models[0].Θ))

fig, (ax2, ax1, ax3) = plt.subplots(nrows=1, ncols=3)
_, ax1_right = ot.plot_powers_and_efficiencies(np.array(shifts) * 100, cold_models, xlabel="Cycle Shift", ax=ax1)[2]
_, ax2_right = ot.plot_powers_and_efficiencies(np.array(shifts) * 100, long_models, xlabel="Cycle Shift", ax=ax2)[2]
_, ax3_right = ot.plot_powers_and_efficiencies(np.array(shifts) * 100, models, xlabel="Cycle Shift", ax=ax3)[2]

ax1_right.sharey(ax2_right)
ax1.sharey(ax2)

ax3_right.sharey(ax1_right)
ax3.sharey(ax1)

ax1.set_title("Cold Shifted")
ax2.set_title("Both Shifted")
fs.export_fig("cycle_shift_power_efficiency_longer_vs_only_cold", y_scaling=.7, x_scaling=2)

ot.plot_multi_powers_and_efficiencies(shifts, [models, long_models, cold_models], ["shifted", "shifted + slower modulation", "slower + only cold shifted"], xlabel=r"Shift $\delta$")
fs.export_fig("shift_comparison", y_scaling=1, x_scaling=2)

ot.plot_bloch_components(off_ax_models[1])

for model in off_ax_models:
    ot.plot_energy(model)
