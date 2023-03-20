import matplotlib.pyplot as plt
import plot_utils as pu
from hiro_models.otto_cycle import OttoEngine
import numpy as np
import figsaver as fs
import hiro_models.model_auxiliary as aux
from typing import Iterable


def plot_power_eff_convergence(models, steady_idx=1):
    f, (a_power, a_efficiency) = plt.subplots(ncols=2)

    a_efficiency.set_yscale("log")
    for model in models:
        Ns = model.power(steady_idx=steady_idx).Ns
        a_power.plot(Ns, model.power(steady_idx=steady_idx).values)
        a_efficiency.plot(Ns, np.abs(model.efficiency(steady_idx=steady_idx).values))

    a_power.set_xlabel("$N$")
    a_power.set_ylabel("$P$")
    a_efficiency.set_xlabel("$N$")
    a_efficiency.set_ylabel("$\eta$")
    return f, (a_power, a_efficiency)


@pu.wrap_plot
def plot_powers_and_efficiencies(x, models, steady_idx=1, ax=None, xlabel=""):
    powers = [-model.power(steady_idx=steady_idx).value for model in models]
    powers_σ = [model.power(steady_idx=steady_idx).σ for model in models]

    system_powers = [
        val_relative_to_steady(
            model,
            -1 * model.system_power().integrate(model.t) * 1 / model.Θ,
            steady_idx=steady_idx,
        )[1].value[-1]
        for model in models
    ]

    system_powers_σ = [
        val_relative_to_steady(
            model,
            -1 * model.system_power().integrate(model.t) * 1 / model.Θ,
            steady_idx=steady_idx,
        )[1].σ[-1]
        for model in models
    ]

    interaction_powers = [
        val_relative_to_steady(
            model,
            -1 * model.interaction_power().sum_baths().integrate(model.t) * 1 / model.Θ,
            steady_idx=steady_idx,
        )[1].value[-1]
        for model in models
    ]

    interaction_powers_σ = [
        val_relative_to_steady(
            model,
            -1 * model.interaction_power().sum_baths().integrate(model.t) * 1 / model.Θ,
            steady_idx=steady_idx,
        )[1].σ[-1]
        for model in models
    ]

    efficiencies = np.array(
        [100 * model.efficiency(steady_idx=steady_idx).value for model in models]
    )

    efficiencies_σ = np.array(
        [100 * model.efficiency(steady_idx=steady_idx).σ for model in models]
    )

    mask = efficiencies > 0
    a2 = ax.twinx()
    ax.errorbar(x, powers, yerr=powers_σ, marker=".", label=r"$\bar{P}$")
    ax.errorbar(
        x,
        system_powers,
        yerr=system_powers_σ,
        marker=".",
        label=r"$\bar{P}_{\mathrm{sys}}$",
    )

    ax.errorbar(
        x,
        interaction_powers,
        yerr=interaction_powers_σ,
        marker=".",
        label=r"$\bar{P}_{\mathrm{int}}$",
    )
    ax.legend()

    lines = a2.errorbar(
        np.asarray(x)[mask],
        efficiencies[mask],
        yerr=efficiencies_σ[mask],
        marker="*",
        color="C4",
        label=r"$\eta$",
    )
    a2.legend(loc="upper left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$-\bar{P}$", color="C0")
    a2.set_ylabel(r"$\eta$", color="C4")


@pu.wrap_plot
def plot_cycle(model: OttoEngine, ax=None):
    assert ax is not None
    ax.plot(
        model.t, model.coupling_operators[0].operator_norm(model.t) * 2, label=r"$L_c$"
    )
    ax.plot(
        model.t, model.coupling_operators[1].operator_norm(model.t) * 2, label=r"$L_h$"
    )

    ax.plot(
        model.t,
        (model.H.operator_norm(model.t)) / model.H.operator_norm(model.τ_compressed),
        label="$H_{\mathrm{sys}}$",
    )

    ax.set_xlim((0, model.Θ))
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"Operator Norm")
    ax.legend()


@pu.wrap_plot
def plot_cycles(
    models: list[OttoEngine],
    ax=None,
    H_for_all=False,
    H=True,
    L_for_all=True,
    bath=None,
    legend=False,
):
    assert ax is not None

    model = models[0]

    if H:
        ax.plot(
            model.t,
            (model.H.operator_norm(model.t))
            / model.H.operator_norm(model.τ_compressed),
            label=f"$H_1$",
        )

    for index, name in enumerate(["c", "h"]):
        if bath is None or bath == index:
            ax.plot(
                model.t,
                model.coupling_operators[index].operator_norm(model.t) * 2,
                label=rf"$L_{{{name},1}}$",
            )

    ax.set_xlim((0, model.Θ))
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"Operator Norm")

    for i, model in enumerate(models[1:]):
        if H and H_for_all:
            ax.plot(
                model.t,
                (model.H.operator_norm(model.t))
                / model.H.operator_norm(model.τ_compressed),
                label=f"$H_1$",
            )

        if L_for_all:
            for index, name in enumerate(["c", "h"]):
                if bath is None or bath == index:
                    ax.plot(
                        model.t,
                        model.coupling_operators[index].operator_norm(model.t) * 2,
                        label=rf"$L_{{{name},{i+2}}}$",
                    )

    legend and ax.legend()


@pu.wrap_plot
def plot_sd_overview(model: OttoEngine, ax=None):
    assert ax is not None

    gaps = model.energy_gaps
    ω = np.linspace(0.0001, gaps[-1] + gaps[0], 1000)

    for ω_i, label, i in zip(gaps, ["Cold", "Hot"], range(len(gaps))):
        lines = ax.plot(
            ω,
            model.full_thermal_spectral_density(i)(ω) * model.bcf_scales[i],
            label=f"{label} $T={model.T[i]}$",
        )

        ax.plot(
            ω,
            model.spectral_density(i)(ω) * model.bcf_scales[i],
            label=f"{label} $T=0$",
            color=pu.lighten_color(lines[0].get_color()),
            linestyle="--",
        )

        ax.plot(
            ω_i,
            model.full_thermal_spectral_density(i)(ω_i) * model.bcf_scales[i],
            marker="o",
            color=lines[0].get_color(),
        )

    # plt.plot(ω, model.full_thermal_spectral_density(1)(ω) * model.bcf_scales[1])
    # plt.plot(
    #     2, model.full_thermal_spectral_density(1)(2) * model.bcf_scales[1], marker="o"
    # )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"Spectral Density")
    ax.legend()


def full_report(model):
    cyc = plot_cycle(model)
    sd = plot_sd_overview(model)

    f, a = plot_energy(model)
    pu.plot_with_σ(model.t, model.total_energy(), ax=a)

    power = model.power()
    η = model.efficiency() * 100

    print(
        fs.tex_value(power.value, err=power.σ, prefix="P="),
    )
    print(
        fs.tex_value(η.value, err=η.σ, prefix=r"\eta="),
    )


def plot_energy(model):
    f, a = pu.plot_energy_overview(
        model,
        strobe_frequency=model.Ω,
        hybrid=True,
        bath_names=["cold", "hot"],
        online=True,
    )

    a.legend()

    return f, a


def integrate_online(model, n, stream_folder=None, **kwargs):
    aux.integrate(
        model,
        n,
        stream_file=("" if stream_folder is None else stream_folder)
        + f"results_{model.hexhash}.fifo",
        analyze=True,
        **kwargs,
    )


def get_sample_count(model):
    try:
        with aux.get_data(model) as d:
            return d.samples

    except:
        return 0


def integrate_online_multi(models, n, *args, increment=1000, **kwargs):
    target = increment

    while target <= n:
        current_target = min([n, target])
        for model in models:
            count = get_sample_count(model)
            if count < current_target:
                integrate_online(model, current_target, *args, **kwargs)

        target += increment


@pu.wrap_plot
def plot_3d_heatmap(models, value_accessor, x_spec, y_spec, normalize=False, ax=None):
    value_dict = {}
    x_labels = set()
    y_labels = set()

    for model in models:
        x_label = x_spec(model)
        y_label = y_spec(model)
        value = value_accessor(model)

        if x_label not in value_dict:
            value_dict[x_label] = {}

        if y_label in value_dict[x_label]:
            raise ValueError(
                f"Dublicate value for model with x={x_label}, y={y_label}."
            )

        value_dict[x_label][y_label] = value_accessor(model)

        x_labels.add(x_label)
        y_labels.add(y_label)

    x_labels = np.sort(list(x_labels))
    y_labels = np.sort(list(y_labels))

    _xx, _yy = np.meshgrid(x_labels, y_labels, indexing="ij")
    x, y = _xx.ravel(), _yy.ravel()

    values = np.fromiter((value_dict[_x][_y] for _x, _y in zip(x, y)), dtype=float)

    dx = x_labels[1] - x_labels[0]
    dy = y_labels[1] - y_labels[0]

    x -= dx / 2
    y -= dy / 2

    normalized_values = abs(values) - abs(values).min()
    normalized_values /= abs(normalized_values).max()

    cmap = plt.get_cmap("plasma")
    colors = [cmap(power) for power in normalized_values]

    ax.bar3d(
        x,
        y,
        np.zeros_like(values),
        dx,
        dy,
        values / abs(values).max() if normalize else values,
        color=colors,
    )
    ax.set_xticks(x_labels)
    ax.set_yticks(y_labels)


def val_relative_to_steady(model, val, steady_idx):
    begin_idx = model.strobe[1][steady_idx]
    return model.t[begin_idx:], (
        val.slice(slice(begin_idx - 1, -1, 1)) - val.slice(begin_idx - 1)
    )


def timings(τ_c, τ_i):
    τ_th = (1 - 2 * τ_c) / 2
    τ_i_on = τ_th - 2 * τ_i
    timings_H = (0, τ_c, τ_c + τ_th, 2 * τ_c + τ_th)
    timings_L_hot = (τ_c, τ_c + τ_i, τ_c + τ_i + τ_i_on, τ_c + 2 * τ_i + τ_i_on)

    timings_L_cold = tuple(time + timings_H[2] for time in timings_L_hot)

    return timings_H, (timings_L_cold, timings_L_hot)


def model_description(model):
    return model.description


def plot_steady_energy_changes(models, steady_idx=2, label_fn=model_description):
    fig, ax = plt.subplots()

    for model in models:
        t, inter = val_relative_to_steady(
            model, model.interaction_power().sum_baths().integrate(model.t), steady_idx
        )
        t, sys = val_relative_to_steady(
            model, model.system_power().sum_baths().integrate(model.t), steady_idx
        )

        pu.plot_with_σ(
            t,
            inter,
            ax=ax,
            label=fr"$W_\mathrm{{int}}$ {label_fn(model)}",
            linestyle="--",
        )
        pu.plot_with_σ(
            t,
            sys,
            ax=ax,
            label=fr"$W_\mathrm{{sys}}$ {label_fn(model)}",
        )

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$W$")
    ax.legend()

    return fig, ax
