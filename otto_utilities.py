import matplotlib.pyplot as plt
import plot_utils as pu
from hiro_models.otto_cycle import OttoEngine, SmoothlyInterpolatdPeriodicMatrix
from hops.util.dynamic_matrix import *
import numpy as np
import figsaver as fs
import hiro_models.model_auxiliary as aux
from typing import Iterable
import qutip as qt
import itertools


def plot_power_eff_convergence(models, steady_idx=2):
    f, (a_power, a_efficiency) = plt.subplots(ncols=2)

    a_efficiency.set_yscale("log")
    for model in models:
        try:
            Ns = model.power(steady_idx=steady_idx).Ns
            a_power.plot(Ns, model.power(steady_idx=steady_idx).values)
            a_efficiency.plot(
                Ns, np.abs(model.efficiency(steady_idx=steady_idx).values)
            )
        except:
            pass

    a_power.set_xlabel("$N$")
    a_power.set_ylabel("$P$")
    a_efficiency.set_xlabel("$N$")
    a_efficiency.set_ylabel("$\eta$")
    return f, (a_power, a_efficiency)


@pu.wrap_plot
def plot_powers_and_efficiencies(x, models, steady_idx=2, ax=None, xlabel=""):
    powers = [-model.power(steady_idx=steady_idx).value for model in models]
    powers_σ = [model.power(steady_idx=steady_idx).σ for model in models]

    ax.axhline(0, color="lightgray")
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
    ax.set_ylabel(r"$\bar{P}$", color="C0")
    a2.set_ylabel(r"$\eta$", color="C4")

    return ax, a2


def plot_multi_powers_and_efficiencies(
    x, multi_models, titles, steady_idx=2, xlabel=""
):
    fig, axs = plt.subplots(nrows=2, ncols=2)
    (efficiency, power, system_power, interaction_power) = axs.flatten()

    markers = itertools.cycle((".", "+", "*", ",", "o"))
    for models, title, marker in zip(multi_models, titles, [".", "^", "*"]):
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
                -1
                * model.interaction_power().sum_baths().integrate(model.t)
                * 1
                / model.Θ,
                steady_idx=steady_idx,
            )[1].value[-1]
            for model in models
        ]

        interaction_powers_σ = [
            val_relative_to_steady(
                model,
                -1
                * model.interaction_power().sum_baths().integrate(model.t)
                * 1
                / model.Θ,
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

        power.plot(x, powers, marker=marker)
        system_power.plot(
            x,
            system_powers,
            marker=marker,
        )

        interaction_power.plot(
            x,
            interaction_powers,
            marker=marker,
        )

        efficiency.plot(
            np.asarray(x)[mask],
            efficiencies[mask],
            marker=marker,
            label=title,
        )

        efficiency.set_title(r"$\eta$")
        power.set_title(r"$\bar{P}$")
        system_power.set_title(
            r"$\bar{P}_{\mathrm{sys}}$",
        )
        interaction_power.set_title(
            r"$\bar{P}_{\mathrm{int}}$",
        )

    fig.supxlabel(xlabel)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    return fig, axs


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
        strobe_data=model.strobe,
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


@pu.wrap_plot
def plot_contour(
    models, value_accessor, x_spec, y_spec, normalize=False, ax=None, levels=None
):
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

    values = (
        np.fromiter((value_dict[_x][_y] for _x, _y in zip(x, y)), dtype=float)
        .reshape(len(x_labels), len(y_labels))
        .T
    )

    normalized_values = abs(values) - abs(values).min()
    normalized_values /= abs(normalized_values).max()

    cont = ax.contourf(
        x_labels,
        y_labels,
        values / abs(values).max() if normalize else values,
        levels=levels,
    )
    ax.set_xticks(x_labels)
    ax.set_yticks(y_labels)
    return cont, (x_labels, y_labels, values)


def get_steady_times(model, steady_idx, shift=0):
    shift_idx = int(1 / model.dt * shift)

    begin_idx = model.strobe[1][steady_idx] - shift_idx
    end_idx = -shift_idx if shift != 0 else -2

    return model.t[begin_idx - 1 : end_idx]


def val_relative_to_steady(model, val, steady_idx, shift=0, absolute=False):
    shift_idx = int(1 / model.dt * shift)
    begin_idx = model.strobe[1][steady_idx] - shift_idx
    end_idx = -shift_idx if shift != 0 else -2

    final_value = val.slice(slice(begin_idx - 1, end_idx, 1))
    if not absolute:
        final_value = final_value - val.slice(begin_idx - 1)

    return (model.t[begin_idx - 1 : end_idx], final_value)


def timings(τ_c, τ_i):
    τ_th = (1 - 2 * τ_c) / 2
    τ_i_on = τ_th - 2 * τ_i
    timings_H = (0, τ_c, τ_c + τ_th, 2 * τ_c + τ_th)
    timings_L_hot = (τ_c, τ_c + τ_i, τ_c + τ_i + τ_i_on, τ_c + 2 * τ_i + τ_i_on)

    timings_L_cold = tuple(time + timings_H[2] for time in timings_L_hot)

    return timings_H, (timings_L_cold, timings_L_hot)


def model_description(model):
    return model.description


@pu.wrap_plot
def plot_steady_energy_changes(
    models,
    steady_idx=2,
    label_fn=model_description,
    bath=None,
    ax=None,
    with_shift=False,
    shift_min_inter=False,
):
    times, inters, systems = [], [], []
    for model in models:
        t, inter = val_relative_to_steady(
            model,
            (
                model.interaction_power().sum_baths()
                if bath is None
                else model.interaction_power().for_bath(bath)
            ).integrate(model.t),
            steady_idx,
            shift=model.L_shift[0] if with_shift else 0,
        )
        t, sys = val_relative_to_steady(
            model,
            model.system_power().sum_baths().integrate(model.t),
            steady_idx,
        )

        inters.append(inter)
        systems.append(sys)
        times.append(t)

    if shift_min_inter:
        for i, inter in enumerate(inters):
            length = len(inter.value)
            inters[i] -= (inter.slice(slice(0, length // 3))).max.value

    for inter, sys, t, model in zip(inters, systems, times, models):
        print(model.L_shift)
        _, _, (l, _) = pu.plot_with_σ(
            t,
            -1 * inter,
            ax=ax,
            label=rf"$W_\mathrm{{int}}$ {label_fn(model)}",
            linestyle="--",
        )
        pu.plot_with_σ(
            t,
            -1 * sys,
            ax=ax,
            label=rf"$W_\mathrm{{sys}}$ {label_fn(model)}",
            color=l[0].get_color(),
        )

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$W$")
    ax.legend()


def add_arrow(line, start_ind=None, direction="right", size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if direction == "right":
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate(
        "",
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="->", color=color),
        size=size,
    )


@pu.wrap_plot
def plot_state_change_diagram(modulation, value, phase_indices, ax=None):
    all_modulation = model.coupling_operators[bath].operator_norm(t)
    phase_indices = (
        np.array(model.coupling_operators[bath]._matrix._timings) * (len(t) - 1)
    ).astype(np.int64)

    modulation_windowed = all_modulation[phase_indices[0] : phase_indices[-1]]
    value_windowed = inter.value[phase_indices[0] : phase_indices[-1]]

    ax.plot(modulation_windowed, value_windowed, linewidth=3, color="cornflowerblue")
    ax.add_collection(
        color_gradient(
            modulation_windowed, value_windowed, "cornflowerblue", "red", linewidth=3
        )
    )

    for begin, end in zip(phase_indices[:-1], phase_indices[1:]):
        ax.scatter(modulation[begin], value[begin], zorder=100, marker=".", s=200)

    for i, index in enumerate(phase_indices[:-1]):
        ax.text(
            modulation[index] + np.max(modulation) * 0.02,
            value[index] + np.max(np.abs(value)) * 0.01,
            str(i + 1),
        )

    return fig, ax


def get_modulation_and_value(model, operator, value, steady_idx=2):
    shift = 0
    timing_operator = operator
    while not isinstance(timing_operator, SmoothlyInterpolatdPeriodicMatrix):
        if isinstance(operator, Shift):
            shift = operator._delta
            timing_operator = operator._matrix

        if isinstance(operator, DynamicMatrixSum):
            timing_operator = operator._left

    t, value = val_relative_to_steady(model, value, steady_idx, absolute=True)

    all_modulation = operator.operator_norm(t)
    all_modulation_deriv = operator.derivative().operator_norm(t)

    timings = np.array(timing_operator._timings)
    phase_indices = (((timings + shift / model.Θ) % 1) * (len(t) - 1)).astype(np.int64)

    values = np.zeros_like(all_modulation)
    np.divide(
        value.value, all_modulation, where=np.abs(all_modulation) > 1e-3, out=values
    ),

    return (
        values,
        all_modulation,
        phase_indices,
    )


def plot_modulation_system_diagram(model, steady_idx):
    fig, ax = plt.subplots()

    t, system = val_relative_to_steady(
        model,
        (model.system_energy().sum_baths()),
        steady_idx,
    )

    modulation = model.H.operator_norm(t)
    ax.plot(modulation, system.value)
    ax.set_xlabel(r"$||H_\mathrm{S}||$")
    ax.set_ylabel(r"$\langle{H_\mathrm{S}}\rangle$")

    return fig, ax


def plot_steady_work_baths(models, steady_idx=2, label_fn=model_description):
    fig, ax = plt.subplots()

    for model in models:
        t, inter_c = val_relative_to_steady(
            model,
            (model.interaction_power().for_bath(0)).integrate(model.t),
            steady_idx,
        )

        t, inter_h = val_relative_to_steady(
            model,
            (model.interaction_power().for_bath(1)).integrate(model.t),
            steady_idx,
        )

        pu.plot_with_σ(
            t,
            inter_c,
            ax=ax,
            label=rf"$W_\mathrm{{int, c}}$ {label_fn(model)}",
        )

        pu.plot_with_σ(
            t,
            inter_h,
            ax=ax,
            label=rf"$W_\mathrm{{int, h}}$ {label_fn(model)}",
            linestyle="--",
        )
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$W$")
    ax.legend()

    return fig, ax


@pu.wrap_plot
def plot_bloch_components(model, ax=None, **kwargs):
    with aux.get_data(model) as data:
        ρ = data.rho_t_accum.mean[:]
        σ_ρ = data.rho_t_accum.ensemble_std[:]

        xs = np.einsum("tij,ji->t", ρ, qt.sigmax().full()).real
        ys = np.einsum("tij,ji->t", ρ, qt.sigmay().full()).real
        zs = np.einsum("tij,ji->t", ρ, qt.sigmaz().full()).real

        ax.plot(
            model.t,
            zs,
            **(dict(label=r"$\langle \sigma_z\rangle$", color="C1") | kwargs),
        )
        ax.plot(
            model.t,
            xs,
            **(dict(label=r"$\langle \sigma_x\rangle$", color="C2") | kwargs),
        )
        ax.plot(
            model.t,
            ys,
            **(dict(label=r"$\langle \sigma_y\rangle$", color="C3") | kwargs),
        )
        ax.legend()
        ax.set_xlabel(r"$\tau$")


@pu.wrap_plot
def plot_energy_deviation(models, ax=None, labels=None):
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\Delta||H||/\max||H||$")

    for i, model in enumerate(models):
        ax.plot(
            model.t,
            abs(model.total_energy_from_power().value - model.total_energy().value)
            / max(abs(model.total_energy_from_power().value)),
            label=labels[i] if labels else None,
        )

    if labels:
        ax.legend()


def max_energy_error(models, steady_idx=None):
    deviations = np.zeros(len(models))

    for i, model in enumerate(models):
        if steady_idx is None:
            deviations[i] = abs(
                model.total_energy_from_power().value - model.total_energy().value
            ).max()
        else:
            deviations[i] = abs(
                val_relative_to_steady(
                    model, model.total_energy_from_power(), steady_idx=steady_idx
                )[1].value
                - val_relative_to_steady(
                    model, model.total_energy(), steady_idx=steady_idx
                )[1].value
            ).max()

    return round(deviations.max() * 100)
