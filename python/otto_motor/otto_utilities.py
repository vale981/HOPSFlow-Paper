import matplotlib.pyplot as plt
import plot_utils as pu
from hiro_models.otto_cycle import OttoEngine
import numpy as np
import figsaver as fs
import hiro_models.model_auxiliary as aux
from typing import Iterable


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
        label="H",
    )

    ax.set_xlim((0, model.Θ))
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"Operator Norm")
    ax.legend()


@pu.wrap_plot
def plot_cycles(models: list[OttoEngine], ax=None, H_for_all=False, L_for_all=True):
    assert ax is not None

    model = models[0]

    ax.plot(
        model.t,
        (model.H.operator_norm(model.t)) / model.H.operator_norm(model.τ_compressed),
        label=f"$H_1$",
    )

    ax.plot(
        model.t,
        model.coupling_operators[0].operator_norm(model.t) * 2,
        label=r"$L_{c,1}$",
    )
    ax.plot(
        model.t,
        model.coupling_operators[1].operator_norm(model.t) * 2,
        label=r"$L_{h,1}$",
    )

    ax.set_xlim((0, model.Θ))
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"Operator Norm")

    for i, model in enumerate(models[1:]):
        if H_for_all:
            ax.plot(
                model.t,
                (model.H.operator_norm(model.t))
                / model.H.operator_norm(model.τ_compressed),
                label=f"$H_1$",
            )

        if L_for_all:
            ax.plot(
                model.t,
                model.coupling_operators[0].operator_norm(model.t) * 2,
                label=rf"$L_{{c,{i+2}}}$",
            )
            ax.plot(
                model.t,
                model.coupling_operators[1].operator_norm(model.t) * 2,
                label=rf"$L_{{h,{i+2}}}$",
            )

    ax.legend()


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
        bath_names=["Cold", "Hot"],
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


def integrate_online_multi(models, n, *args, increment=1000, **kwargs):
    target = increment

    samples = []
    for model in models:
        try:
            with aux.get_data(model, *args, **kwargs) as d:
                samples.append(d.samples)

        except:
            samples.append(0)

    while target < (n + target):
        for model, s in zip(models, samples):
            if s < target:
                integrate_online(model, min([n, target]), *args, **kwargs)

        target += increment


def plot_3d_heatmap(models, value_accessor, x_spec, y_spec):
    f, _ = plt.subplots()

    ax1 = plt.gcf().add_subplot(111, projection="3d")

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

    ax1.bar3d(x, y, np.zeros_like(values), dx, dy, values, color=colors)
    ax1.set_xticks(x_labels)
    ax1.set_yticks(y_labels)

    return f, ax1
