import plot_utils as pu
from hiro_models.one_qubit_model import StocProcTolerances
from hiro_models.otto_cycle import OttoEngine, get_energy_gap
import hiro_models.model_auxiliary as aux
import numpy as np
import qutip as qt
import utilities as ut
import stocproc
import matplotlib.pyplot as plt
import otto_utilities as ot
import shift_cycle as sc
import ray
import figsaver as fs
ray.shutdown()

#ray.init(address='auto')
ray.init()
from hops.util.logging_setup import logging_setup
import logging
logging_setup(logging.INFO)

ot.plot_cycle(baseline)
fs.export_fig("cycle_prototype", y_scaling=.7)

system = [
    np.divide(σ, val, where=val>0).mean()
    for σ, val in zip(baseline.system_energy().σs, baseline.system_energy().values)
]
flow = [
    np.divide(σ, val, where=val>0).mean()
    for σ, val in zip(baseline.bath_energy_flow().σs, baseline.bath_energy_flow().values)
]
plt.plot(system)
plt.plot(flow)

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
for model in [baseline]:
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

import pickle

def save_data(model, name):
    data = [
        {
            "name": f"bath_modulation_interaction_{bath_name}",
            "xlabel": rf"$||L_{bath_name}(t)||$",
            "ylabel": r"$\langle{H_\mathrm{I}}\rangle$",
            "args": ot.get_modulation_and_value(
                model,
                model.coupling_operators[bath],
                model.interaction_energy().for_bath(bath),
            ),
        }
        for bath, bath_name in zip([0, 1], ["c", "h"])
    ]  + [
        {
            "name": f"system_modulation_system_energy",
            "xlabel": r"$||H_\mathrm{S}||$",
            "ylabel": r"$\langle{H_\mathrm{S}}\rangle$",
            "args": ot.get_modulation_and_value(
                model,
                model.H,
                model.system_energy(),
                steady_idx=2
            ),
        }
    ]

    with open(f"data/pv_{name}.pickle", "wb") as file:
        pickle.dump(data, file)





# vals = ot.get_modulation_and_value(model, model.coupling_operators[0], model.interaction_energy().for_bath(0))
# plot_modulation_interaction_diagram(*vals)

save_data(baseline, "baseline")

for model in models:
  print(model.power(steady_idx=2).value / baseline.power(steady_idx=2).value, model.efficiency(steady_idx=2).value)

ot.plot_power_eff_convergence(models)
fs.export_fig("cycle_shift_convergence", x_scaling=2, y_scaling=.7)

ot.plot_powers_and_efficiencies(np.array(shifts) * 100, models, xlabel="Cycle Shift")
fs.export_fig("cycle_shift_power_efficiency", y_scaling=.7, x_scaling=1)

best_shift = shifts[np.argmax([-model.power(steady_idx=2).value for model in models])]
best_shift_model = sc.make_model(best_shift, best_shift)
best_shift

fig, ax =ot.plot_steady_energy_changes([baseline, best_shift_model], 2, label_fn=lambda m: ("baseline" if m.hexhash == baseline.hexhash else "shifted"))
ax.legend(loc="lower left")
fs.export_fig("shift_energy_change", y_scaling=.7)

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

f, a = plt.subplots()
a.axhline(best_shift_model.system_energy().value[np.argmin(abs(best_shift_model.t - model.Θ * 2))], color="gray", linestyle="--")
r = pu.plot_with_σ(
    best_shift_model.t, best_shift_model.interaction_energy().sum_baths(), ax=a,
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


a.plot(
    best_shift_model.t, best_shift_model.system.operator_norm(best_shift_model.t) / 5,
    label="system modulation"
)

a.plot(
    best_shift_model.t,
    best_shift_model.coupling_operators[1].operator_norm(best_shift_model.t) / 5,
    label="hot bath modulation",
)
# a.plot(best_shift_model.t, best_shift_model.coupling_operators[1].operator_norm(best_shift_model.t) / 5)
a.set_xlim((model.Θ * 2, model.Θ * 2 + 20))

a.set_ylim((-.21, .45))
a.set_xlabel(r"$\tau$")
a.legend(loc="upper right", fontsize="x-small")
fs.export_fig("cold_bath_decoupling", y_scaling=.6)

f, a = plt.subplots()
a.axhline(baseline.system_energy().value[np.argmin(abs(baseline.t - model.Θ * 2))], color="gray", linestyle="--")
r = pu.plot_with_σ(
    baseline.t, baseline.interaction_energy().sum_baths(), ax=a,
    label=r"$\langle H_\mathrm{inter}\rangle$"
)
pu.plot_with_σ(
    baseline.t, baseline.system_energy(), ax=a, label=r"$\langle H_\mathrm{sys}\rangle$"
)
# a.plot(baseline.t, baseline.H(baseline.t)[:, 0,0])
a.plot(
    baseline.t,
    baseline.coupling_operators[0].operator_norm(baseline.t) / 5,
    label="cold bath modulation",
)

a.plot(
      baseline.t, baseline.system.operator_norm(baseline.t) / 5,
      label="system modulation"
  )
a.plot(
    baseline.t,
    baseline.coupling_operators[1].operator_norm(baseline.t) / 5,
    label="hot bath modulation",
)

# a.plot(baseline.t, baseline.coupling_operators[1].operator_norm(baseline.t) / 5)
a.set_xlim((model.Θ * 2-5, model.Θ * 2 + 13))

a.set_ylim((-.21, .45))
a.set_xlabel(r"$\tau$")
a.legend(loc="upper right", fontsize="x-small")
#fs.export_fig("cold_bath_decoupling", y_scaling=.6)
fs.export_fig("cold_bath_decoupling_baseline", y_scaling=.6)

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

best_long_idx = np.argmax([-model.power(steady_idx=2).value for model in long_models])
best_long_shift = shifts[best_long_idx]
best_long_shift_model = long_models[best_long_idx]
best_long_shift

fig, ax =ot.plot_steady_energy_changes([best_long_shift_model, best_shift_model], 2, label_fn=lambda m: ("long" if m.hexhash == best_long_shift_model.hexhash else "short"))
ax.legend(loc="lower left")

fs.export_fig("long_short_energy_change", y_scaling=.7)

best_long_model = best_long_shift_model

flow_long = -1*best_long_model.bath_energy_flow().for_bath(0)
power_long = best_long_model.interaction_power().for_bath(0)

flow_short = -1*best_shift_model.bath_energy_flow().for_bath(0)
power_short = best_shift_model.interaction_power().for_bath(0)

plt.plot(best_shift_model.t, flow_short.value, label="fast coupling")
plt.plot(best_shift_model.t, flow_long.value, label="slow coupling")
plt.plot(best_shift_model.t, power_short.value, linestyle="--", color="C0")
plt.plot(best_shift_model.t, power_long.value, linestyle="--",  color="C1")
plt.xlim((2*best_long_model.Θ-5, 2*best_long_model.Θ+12))
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

#aux.import_results(other_data_path="taurus/.data", other_results_path="taurus/results", models_to_import=cold_models)

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
ax3.set_title("Fast Modulation")
fs.export_fig("cycle_shift_power_efficiency_longer_vs_only_cold", y_scaling=.7, x_scaling=2.5)

ot.plot_multi_powers_and_efficiencies(shifts, [models, long_models, cold_models], ["shifted", "shifted + slower modulation", "slower + only cold shifted"], xlabel=r"Shift $\delta$")
fs.export_fig("shift_comparison", y_scaling=1, x_scaling=2)

best_cold_shift = shifts[np.argmax([-model.power(steady_idx=2).value for model in cold_models])]
best_cold_model = sc.make_model(best_cold_shift, best_cold_shift, switch_t=6., only_cold=True)
best_cold_shift

fig, ax =ot.plot_steady_energy_changes([best_cold_model], 2, label_fn=lambda m: "")
ax.legend(loc="lower left")

fs.export_fig("steady_energy_dynamics_slow_only_cold_shifted", y_scaling=.7)

import matplotlib.pyplot as plt

names = {
    baseline.hexhash: "Otto-Cycle",
    best_shift_model.hexhash: "Shifted Strokes",
    best_long_model.hexhash: "Slow Modulation + Both Strokes Shifted",
    best_cold_model.hexhash: "Slow Modulation + Cold Stroke Shifted",
}

# Increase the size of the plot
fig, ax = plt.subplots(figsize=(15, 6))

# Assuming ot.plot_steady_energy_changes returns a Line2D object for each line
lines = ot.plot_steady_energy_changes(
    [baseline, best_shift_model, best_long_model, best_cold_model],
    2,
    label_fn=lambda m: names[m.hexhash],
    ax=ax,
    shift_min_inter = False
)

# Move the legend outside the plot
ax.legend(loc="lower left", bbox_to_anchor=(1, 0.5), fontsize='small')


# Adjust layout to make room for the legend
plt.tight_layout()

fs.export_fig("steady_energy_dynamics_all_models", y_scaling=.7)

def get_modulations(model, label):
    t = ot.get_steady_times(model, 2)
    t = np.linspace(t.min(), t.max(), 10000)
    return {
        "system": model.H.operator_norm(t),
        "cold": model.coupling_operators[0].operator_norm(t) * 2,
        "hot": model.coupling_operators[1].operator_norm(t) * 2,
        "time": t,
        "timings": t[0] +  np.array(model.timings_H) * model.Θ,
        "timings_hot": t[0] +  (np.array(model.timings_L[1]) + model.L_shift[1]) * model.Θ,
        "label": label
    }

modulations = {
    "baseline": get_modulations(baseline, "Otto-like\ncycle"),
    "best_shift_model": get_modulations(best_shift_model, "shifted\nstrokes"),
    "best_long_model": get_modulations(best_long_model, "shifted strokes,\nslow mod."),
    "best_cold_model": get_modulations(best_cold_model, "cold shifted,\nslow mod.")
}

import pickle
with open("data/modulations.pickle", "wb") as f:
    pickle.dump(modulations, f)

ot.plot_energy_deviation([best_cold_model, baseline])

ints = baseline.interaction_energy().for_bath(1).value
powers = baseline.interaction_power().for_bath(1).value
mods = baseline.coupling_operators[1].operator_norm(baseline.t)
mods_deriv = baseline.coupling_operators[1].derivative().operator_norm(baseline.t)
raw_interaction = np.divide(ints, mods, where=abs(mods) > 1e-2)
raw_interaction_from_power = -abs(np.divide(powers, mods_deriv, where=abs(mods_deriv) > 1e-3))
plt.plot(baseline.t, raw_interaction_from_power)
plt.plot(baseline.t, raw_interaction)
plt.yscale("symlog")

import scipy

plt.plot(mods, raw_interaction)
plt.plot(mods, raw_interaction_from_power)
scipy.integrate.simpson(raw_interaction, mods)

baseline.interaction_power().for_bath(1).integrate(baseline.t).value[-1]

plt.plot(best_cold_model.t, best_cold_model.coupling_operators[0].operator_norm(best_cold_model.t))
plt.plot(best_cold_model.t, best_cold_model.H.operator_norm(best_cold_model.t))

save_data(best_cold_model, "slow_shifted")

aux.import_results(other_data_path="taurus/.data_oa", other_results_path="taurus/results")

for (i, model), weight in zip(enumerate(off_ax_models), weights):
    f, a = ot.plot_bloch_components(model)
    #ot.plot_bloch_components(off_ax_models[i+2], ax=a, linestyle="--", label=None)

    a.set_title(rf"$r_y={weight}$")
    fs.export_fig(f"bloch_expectation_offaxis_{weight}", y_scaling=.7)

np.array(weights) / np.sqrt(1 + np.array(weights) ** 2)

baselines = [baseline] * 2 + [long_baseline] * 2
for model, ref in zip(off_ax_models, baselines):
    print(model.power(steady_idx=2).value / ref.power(steady_idx=2).value, model.efficiency(steady_idx=2).value / ref.efficiency(steady_idx=2).value)

for (i, model), weight in zip(enumerate(off_ax_models), weights):
    f, a = ot.plot_energy(model)
    a.set_title(rf"$r_y={weight}$")
    fs.export_fig(f"full_energy_offaxis_{weight}", x_scaling=2, y_scaling=1)

fig, axs = plt.subplots(ncols=2)
(ax_full, ax) = axs

for ax in axs:
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\Delta X$")

for (i, model) in enumerate([off_ax_models[0], baseline]):
    for j, (val, label) in enumerate(zip([
        model.total_energy_from_power(),
        model.system_energy(),
        model.interaction_energy().sum_baths(),
        model.bath_energy().for_bath(0),
        model.bath_energy().for_bath(1),
    ], ["Total", "System", "Interaction", "Cold Bath", "Hot Bath"])):
        linestyle = "dashed" if model == baseline else None
        pu.plot_with_σ(model.t[:1000], val.slice(slice(0, 1000, 1)), ax=ax_full, linestyle=linestyle, color=f"C{j}")
        t, steady_total = ot.val_relative_to_steady(model, val, steady_idx=2)
        pu.plot_with_σ(t, steady_total, ax=ax, label=label if model != baseline else None, linestyle=linestyle, color=f"C{j}")

ax.legend()
fs.export_fig(f"energy_change_off_axis", x_scaling=2, y_scaling=0.7)

τs = rot_models[0].t
#plt.plot(τs, np.einsum('tij,ij->t', rot_models[0].H(τs), qt.sigmay().full()).real)
# plt.plot(τs, abs(rot_models[0].H(τs)[:, 0, 0]))
# plt.plot(τs, abs(rot_models[0].H(τs)[:, 0, 1]))
# plt.plot(τs, abs(rot_models[0].H.operator_norm(τs)))
H = rot_models[0].H
plt.plot(τs, list(map(lambda t: get_energy_gap(H(t)), τs)), color="black")

for model in rot_models:
    print(model.energy_gaps[1] - model.energy_gaps[0])

aux.import_results(other_data_path="taurus/.data", other_results_path="taurus/results", models_to_import=rot_models)

for (i, model), weight in zip(enumerate(rot_models), weights):
    f, a = ot.plot_bloch_components(model)
    #ot.plot_bloch_components(off_ax_models[i+2], ax=a, linestyle="--", label=None)

    a.set_title(rf"$r_x={weight}$")
    fs.export_fig(f"bloch_expectation_rot_{weight}", y_scaling=.7)

for (i, model), weight in zip(enumerate(rot_models), weights):
    f, a = ot.plot_energy(model)
    a.set_title(rf"$r_y={weight}$")
    fs.export_fig(f"full_energy_rot_{weight}", x_scaling=2, y_scaling=1)

for model in rot_models:
    print(model.power(steady_idx=2).value / baseline.power(steady_idx=2).value, model.efficiency(steady_idx=2).value / baseline.efficiency(steady_idx=2).value)

fig, axs = plt.subplots(ncols=2)
(ax_full, ax) = axs

for ax in axs:
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\Delta X$")

for (i, model) in enumerate([*rot_models[1:], baseline]):
    for j, (val, label) in enumerate(zip([
        model.total_energy_from_power(),
        model.system_energy(),
        model.interaction_energy().sum_baths(),
        model.bath_energy().for_bath(0),
        model.bath_energy().for_bath(1),
    ], ["Total", "System", "Interaction", "Cold Bath", "Hot Bath"])):
        linestyle = "dashed" if model == baseline else None
        pu.plot_with_σ(model.t[:1000], val.slice(slice(0, 1000, 1)), ax=ax_full, linestyle=linestyle, color=f"C{j}")
        t, steady_total = ot.val_relative_to_steady(model, val, steady_idx=2)
        pu.plot_with_σ(t, steady_total, ax=ax, label=label if model != baseline else None, linestyle=linestyle, color=f"C{j}")

ax.legend()
fs.export_fig(f"energy_change_rot", x_scaling=2, y_scaling=0.7)
