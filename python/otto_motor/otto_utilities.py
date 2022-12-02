import matplotlib.pyplot as plt
import plot_utils as pu
from hiro_models.otto_cycle import OttoEngine
import numpy as np
import figsaver as fs
import hiro_models.model_auxiliary as aux


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
def plot_sd_overview(model: OttoEngine, ax=None):
    assert ax is not None

    gaps = model.energy_gaps
    ω = np.linspace(0.001, gaps[-1] + gaps[0], 1000)

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


def integrate_online(model, n):
    aux.integrate(model, n, stream_file=f"results_{model.hexhash}.fifo", analyze=True)
