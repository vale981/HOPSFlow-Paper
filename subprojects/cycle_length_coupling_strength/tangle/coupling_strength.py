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
#plt.rcParams['figure.figsize'] = (12,4)

def make_model(Θ, δ):
    (p_H, p_L) = ot.timings(.06, .06)
    return OttoEngine(
          δ=[δ, δ],
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
          num_cycles=3 if δ >= .25 else 5,
          Θ=Θ,
          dt=0.001,
          timings_H=p_H,
          timings_L=p_L,
          streaming_mode=True,
          shift_to_resonance=(False, False),
          L_shift=(0, 0),
      )

δs = [.2] + [round(δ, 3) for δ in np.linspace(.3, .7, 5)]
Θs = [round(Θ, 3) for Θ in np.linspace(20, 80, 5)][1:]
δs, Θs

import itertools
models = [make_model(Θ, δ) for Θ, δ, in itertools.product(Θs, δs)]

ot.integrate_online_multi(models, 50_000, increment=10_000, analyze_kwargs=dict(every=10_000))

aux.import_results(other_data_path="taurus/.data", other_results_path="taurus/results", models_to_import=models)

ot.max_energy_error(models), ot.max_energy_error(models, steady_idx=2)

ot.plot_energy(models[5])

f_power = plt.figure()
a_power = f_power.add_subplot(1, 1, 1, projection="3d")
f_work = plt.figure()
a_work = f_work.add_subplot(1, 1, 1, projection="3d")
f_efficiency = plt.figure()
a_efficiency = f_efficiency.add_subplot(1, 1, 1, projection="3d")
f_mean_inter_power = plt.figure()
a_mean_inter_power = f_mean_inter_power.add_subplot(1, 1, 1, projection="3d")
f_mean_system_power = plt.figure()
a_mean_system_power = f_mean_system_power.add_subplot(1, 1, 1, projection="3d")

for ax in [a_power, a_efficiency, a_work, a_mean_inter_power, a_mean_system_power]:
    ax.set_box_aspect(aspect=None, zoom=.7)
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\Theta$")
    ax.xaxis.labelpad = 10
    ax.view_init(elev=30.0, azim=-29, roll=0)


ot.plot_3d_heatmap(
    models,
    lambda model: np.clip(
        np.nan_to_num(model.efficiency(steady_idx=-2).value * 100), 0, np.inf
    ),
    lambda model: model.δ[0],
    lambda model: model.Θ,
    ax=a_efficiency,
)
a_efficiency.set_zlabel(r"$\eta$")

ot.plot_3d_heatmap(
    models,
    lambda model: np.clip(-model.power(steady_idx=-2).value * 1000, 0, np.inf),
    lambda model: model.δ[0],
    lambda model: model.Θ,
    ax=a_power,
)
a_power.set_zlabel(r"$\bar{P}/10^{-3}$")
a_power.zaxis.labelpad = 8

ot.plot_3d_heatmap(
    models,
    lambda model: np.clip(
        ot.val_relative_to_steady(model, model.interaction_power().sum_baths(), 2)[
            1
        ].mean.value
        * 1000,
        0,
        np.inf,
    ),
    lambda model: model.δ[0],
    lambda model: model.Θ,
    ax=a_mean_inter_power,
)
a_mean_inter_power.set_zlabel(r"$-\bar{P}_\mathrm{int}/10^{-3}$")
a_mean_inter_power.zaxis.labelpad = 8
a_mean_inter_power.view_init(elev=30.0, azim=110, roll=0)

ot.plot_3d_heatmap(
    models,
    lambda model: np.clip(
        -ot.val_relative_to_steady(model, model.system_power().sum_baths(), 2)[
            1
        ].mean.value
        * 1000,
        0,
        np.inf,
    ),
    lambda model: model.δ[0],
    lambda model: model.Θ,
    ax=a_mean_system_power,
)
a_mean_system_power.set_zlabel(r"$\bar{P}_\mathrm{sys}/10^{-3}$")
a_mean_system_power.zaxis.labelpad = 8

ot.plot_3d_heatmap(
    models,
    lambda model: np.clip(-model.power(steady_idx=-2).value * model.Θ, 0, np.inf),
    lambda model: model.δ[0],
    lambda model: model.Θ,
    ax=a_work,
)
a_work.set_zlabel(r"$W$")
a_work.zaxis.labelpad = 8


plt.tight_layout()

fs.export_fig("coupling_speed_scan_power", x_scaling=1, y_scaling=1, fig=f_power)
fs.export_fig("coupling_speed_scan_work", x_scaling=1, y_scaling=1, fig=f_work)
fs.export_fig(
    "coupling_speed_scan_efficiency", x_scaling=1, y_scaling=1, fig=f_efficiency
)
fs.export_fig(
    "coupling_speed_scan_interpower", x_scaling=1, y_scaling=1, fig=f_mean_inter_power
)
fs.export_fig(
    "coupling_speed_scan_syspower", x_scaling=1, y_scaling=1, fig=f_mean_system_power
)

f_mean_system_power = plt.figure()
a_mean_system_power = f_mean_system_power.add_subplot(1, 1, 1)

(_, _, (c_mean_sytem_power, data_mean_system_power)) = ot.plot_contour(
      models,
      lambda model:
          -ot.val_relative_to_steady(model, model.system_power().sum_baths(), 2)[
              1
          ].mean.value,
      lambda model: model.δ[0],
      lambda model: model.Θ,
      ax=a_mean_system_power,
  )
a_mean_system_power.set_title(r"$\bar{P}_\mathrm{sys}/\Omega^2$")

f_power = plt.figure()
a_power = f_power.add_subplot(1, 1, 1)
f_work = plt.figure()
a_work = f_work.add_subplot(1, 1, 1)
f_efficiency = plt.figure()
a_efficiency = f_efficiency.add_subplot(1, 1, 1)
f_mean_inter_power = plt.figure()
a_mean_inter_power = f_mean_inter_power.add_subplot(1, 1, 1)
f_mean_system_power = plt.figure()
a_mean_system_power = f_mean_system_power.add_subplot(1, 1, 1)

axs = [a_power, a_efficiency, a_work, a_mean_inter_power, a_mean_system_power]
figs = [f_power, f_efficiency, f_work, f_mean_inter_power, f_mean_system_power]
for ax in axs:
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\Theta$")


(_, _, (c_efficiency, data_efficiency)) = ot.plot_contour(
    models,
    lambda model: np.clip(
        np.nan_to_num(model.efficiency(steady_idx=-2).value * 100), 0, np.inf
    ),
    lambda model: model.δ[0],
    lambda model: model.Θ,
    ax=a_efficiency,
)
a_efficiency.set_title(r"$\eta$")

(_, _, (c_power, data_power)) =ot.plot_contour(
    models,
    lambda model: np.clip(-model.power(steady_idx=-2).value, 0, np.inf),
    lambda model: model.δ[0],
    lambda model: model.Θ,
    ax=a_power,
)
a_power.set_title(r"$\bar{P}/\Omega^2$")

(_, _, (c_mean_inter_power, data_mean_inter_power)) = ot.plot_contour(
    models,
    lambda model: np.clip(
        ot.val_relative_to_steady(model, model.interaction_power().sum_baths(), 2)[
            1
        ].mean.value,
        0,
        np.inf,
    ),
    lambda model: model.δ[0],
    lambda model: model.Θ,
    ax=a_mean_inter_power,
)
a_mean_inter_power.set_title(r"$-\bar{P}_\mathrm{int}/\Omega^2$")

(_, _, (c_mean_system_power, data_mean_system_power)) = ot.plot_contour(
    models,
    lambda model:
        -ot.val_relative_to_steady(model, model.system_power().sum_baths(), 2)[
            1
        ].mean.value,
    lambda model: model.δ[0],
    lambda model: model.Θ,
    ax=a_mean_system_power,
)
a_mean_system_power.set_title(r"$\bar{P}_\mathrm{sys}/\Omega^2$")

(_, _, (c_work, data_work)) = ot.plot_contour(
    models,
    lambda model: np.clip(-model.power(steady_idx=-2).value * model.Θ, 0, np.inf),
    lambda model: model.δ[0],
    lambda model: model.Θ,
    ax=a_work,
)
a_work.set_title(r"$W/\Omega$")


plt.tight_layout()
contours = [c_power, c_efficiency, c_work, c_mean_inter_power, c_mean_system_power]
datas = [data_power, data_efficiency, data_work, data_mean_inter_power, data_mean_system_power]

for fig, contour in zip(figs, contours):
    fig.colorbar(contour)

fs.export_fig("coupling_speed_scan_power_contour", x_scaling=1, y_scaling=1, fig=f_power, data=data_power)
fs.export_fig("coupling_speed_scan_work_contour", x_scaling=1, y_scaling=1, fig=f_work, data=data_work)
fs.export_fig(
    "coupling_speed_scan_efficiency_contour", x_scaling=1, y_scaling=1, fig=f_efficiency, data=data_efficiency
)
fs.export_fig(
    "coupling_speed_scan_interpower_contour", x_scaling=1, y_scaling=1, fig=f_mean_inter_power, data=data_mean_inter_power
)
fs.export_fig(
    "coupling_speed_scan_syspower_contour", x_scaling=1, y_scaling=1, fig=f_mean_system_power, data=data_mean_system_power
)

f = plt.figure()
a_power = f.add_subplot(121, projection="3d")
a_efficiency = f.add_subplot(122, projection="3d")
for ax in [a_power, a_efficiency]:
    ax.set_box_aspect(aspect=None, zoom=0.7)
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\Theta$")

ot.plot_3d_heatmap(
    models,
    lambda model: np.divide(np.abs(model.power(steady_idx=-2).σ), np.abs(model.power(steady_idx=-2).value)),
    lambda model: model.δ[0],
    lambda model: model.Θ,
    ax=a_power,
)
a_power.set_zlabel(r"$\sigma_P/|P|$")


ot.plot_3d_heatmap(
    models,
    lambda model: np.divide(np.clip(np.nan_to_num(model.efficiency(steady_idx=-2).σ * 100), 0, np.inf), np.abs(model.efficiency(steady_idx=-2).value * 100)),
    lambda model: model.δ[0],
    lambda model: model.Θ,
    ax=a_efficiency,
)
a_efficiency.set_zlabel(r"$\sigma_\eta/|\eta|$")
fs.export_fig("coupling_speed_scan_power_efficiency_uncertainty")

ot.plot_energy(weak_coupling_model)

weak_coupling_model.power(steady_idx=-2).value

weak_coupling_model.efficiency(steady_idx=-2).value

weak_coupling_model.strobe
