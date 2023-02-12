from speed_coupling_scan import *

taurus_path = "taurus"
from hiro_models.model_auxiliary import import_results

import_results(
    other_data_path="./taurus/.data",
    other_results_path="./taurus/results",
    interactive=False,
    models_to_import=models,
    force=True,
)

f, a = plt.subplots()

for model in models:
    Δs = (model.steady_index(observable=model.system_energy()))
    #Δ = (model.steady_index(observable=model.total_power(), fraction=.7))
    # for Δ in Δs[2:]:
    #     pu.plot_with_σ(model.t[:model.strobe[1][1]], Δ, ax=a)
    #plt.plot(Δ)
    print(Δs)

    try:
        # pu.plot_with_σ(model.t, model.total_energy_from_power().sum_baths(), ax=a, label=fr"$\delta={model.δ[0]}$, $\tau_I={model.timings_L[0][1] - model.timings_L[0][0]:.3}$")
        # pu.plot_with_σ(model.t, model.total_energy().sum_baths(), ax=a, label=fr"$\delta={model.δ[0]}$, $\tau_I={model.timings_L[0][1] - model.timings_L[0][0]:.3}$")
        pu.plot_with_σ(model.t, model.total_energy_from_power(), ax=a, label=fr"$\delta={model.δ[0]}$, $\tau_I={model.timings_L[0][1] - model.timings_L[0][0]:.3}$")
        pu.plot_with_σ(model.t, model.total_energy(), ax=a, label=fr"$\delta={model.δ[0]}$, $\tau_I={model.timings_L[0][1] - model.timings_L[0][0]:.3}$")
        print(model.system_energy().N)
        print(model.system_power().N)
        print(model.interaction_power().N)
    except:
        pass
a.legend()

f, a =ot.plot_3d_heatmap(models, lambda model: -model.power(fraction=.3).value, lambda model: model.δ[0], lambda model: model.timings_L[0][1] - model.timings_L[0][0])
a.set_xlabel(r"$\delta$")
a.set_ylabel(r"$\tau_I$")
a.set_zlabel(r"$P$")

f, a = plt.subplots()

for model in models:
    try:
        power = model.power(fraction=.5)
        a.plot(power.Ns, power.values, label=fr"$\delta={model.δ[0]}$, $\tau_I={model.timings_L[0][1] - model.timings_L[0][0]:.3}$")
    except:
        pass
a.legend()

from speed_coupling_scan import *

taurus_path = "taurus"
  from hiro_models.model_auxiliary import import_results

  import_results(
      other_data_path="./taurus/.data",
      other_results_path="./taurus/results",
      interactive=False,
      models_to_import=models,
#      force=True,
  )

f, a = plt.subplots()

for model in models:
    Δs = (model.steady_index(observable=model.system_energy()))
    #Δ = (model.steady_index(observable=model.total_power(), fraction=.7))
    # for Δ in Δs[2:]:
    #     pu.plot_with_σ(model.t[:model.strobe[1][1]], Δ, ax=a)
    #plt.plot(Δ)
    print(Δs)

    try:
        # pu.plot_with_σ(model.t, model.total_energy_from_power().sum_baths(), ax=a, label=fr"$\delta={model.δ[0]}$, $\tau_I={model.timings_L[0][1] - model.timings_L[0][0]:.3}$")
        # pu.plot_with_σ(model.t, model.total_energy().sum_baths(), ax=a, label=fr"$\delta={model.δ[0]}$, $\tau_I={model.timings_L[0][1] - model.timings_L[0][0]:.3}$")
        pu.plot_with_σ(model.t, model.total_energy_from_power(), ax=a, label=fr"$\delta={model.δ[0]}$, $\tau_I={model.timings_L[0][1] - model.timings_L[0][0]:.3}$")
        pu.plot_with_σ(model.t, model.total_energy(), ax=a, label=fr"$\delta={model.δ[0]}$, $\tau_I={model.timings_L[0][1] - model.timings_L[0][0]:.3}$")
        print(model.system_energy().N)
        print(model.system_power().N)
        print(model.interaction_power().N)
    except:
        pass
a.legend()

f = plt.figure()
a_power = f.add_subplot(121, projection='3d')
a_efficiency = f.add_subplot(122, projection='3d')

ot.plot_3d_heatmap(
    models,
    lambda model: -model.power(fraction=0.5).value,
    lambda model: model.δ[0],
    lambda model: model.timings_L[0][1] - model.timings_L[0][0],
    normalize=True,
    ax=a_power,
)
a_power.set_xlabel(r"$\delta$")
a_power.set_ylabel(r"$\tau_I$")
a_power.set_zlabel(r"$P$ (normalized)")

ot.plot_3d_heatmap(
    models,
    lambda model: model.efficiency(fraction=0.5).value,
    lambda model: model.δ[0],
    lambda model: model.timings_L[0][1] - model.timings_L[0][0],
    ax=a_efficiency,
)
a_efficiency.set_xlabel(r"$\delta$")
a_efficiency.set_ylabel(r"$\tau_I$")
a_efficiency.set_zlabel(r"$\eta$")

fs.export_fig("coupling_speed_scan", fig=f)
f

f, a = plt.subplots()

for model in models:
    try:
        power = model.power(fraction=.5)
        a.plot(power.Ns, power.values, label=fr"$\delta={model.δ[0]}$, $\tau_I={model.timings_L[0][1] - model.timings_L[0][0]:.3}$")
    except:
        pass
a.legend()
