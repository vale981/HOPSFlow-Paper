import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from contextlib import contextmanager
import utilities as ut
from hopsflow.util import EnsembleValue
from hops.util.utilities import (
    relative_entropy,
    relative_entropy_single,
    entropy,
    trace_distance,
)
from matplotlib.gridspec import GridSpec

try:
    import hiro_models.model_auxiliary as aux
except:
    aux = None


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


###############################################################################
#                                  Plot Porn                                  #
###############################################################################


def wrap_plot(f):
    def wrapped(*args, ax=None, setup_function=plt.subplots, **kwargs):
        fig = None
        if not ax:
            fig, ax = setup_function()

        ret_val = f(*args, ax=ax, **kwargs)
        return (fig, ax, ret_val) if ret_val else (fig, ax)

    return wrapped


def get_figsize(width="thesis", fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "thesis":
        width_pt = 330.62111
    elif width == "poster":
        width_pt = 957.13617
    elif width == "beamer":
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


# def get_figsize(
#     columnwidth: float,
#     wf: float = 0.5,
#     hf: float = (5.0 ** 0.5 - 1.0) / 2.0,
# ) -> tuple[float, float]:
#     """
#     :param wf: Width fraction in columnwidth units.
#     :param hf: Weight fraction in columnwidth units.

#         Set by default to golden ratio.

#     :param columnwidth: width of the column in latex.

#         Get this from LaTeX using \showthe\columnwidth

#     :returns: The ``[fig_width,fig_height]`` that should be given to
#               matplotlib.
#     """
#     fig_width_pt = columnwidth * wf
#     inches_per_pt = 1.0 / 72.27  # Convert pt to inch
#     fig_width = fig_width_pt * inches_per_pt  # width in inches
#     fig_height = fig_width * hf  # height in inches
#     return [fig_width, fig_height]


@wrap_plot
def plot_complex(x, y, *args, ax=None, label="", absolute=False, **kwargs):
    label = label + ", " if (len(label) > 0) else ""
    ax.plot(
        x,
        abs(y.real) if absolute else y.real,
        *args,
        label=f"{label}{'absolute' if absolute else ''} real part",
        **kwargs,
    )
    ax.plot(
        x,
        abs(y.imag) if absolute else y.imag,
        *args,
        label=f"{label}{'absolute' if absolute else ''} imag part",
        **kwargs,
    )
    ax.legend()


@wrap_plot
def plot_convergence(
    x,
    y,
    reference=None,
    ax=None,
    label="",
    slice=None,
    linestyle="-",
    bath=None,
):
    if bath is not None:
        y = y.for_bath(bath)

    label = label + ", " if (len(label) > 0) else ""
    slice = (0, -1) if not slice else slice
    y_final = y[-1]

    for i in range(len(y) - 1 if reference is None else len(y)):
        current_value = y[i]
        consistency = current_value.consistency(
            y_final if reference is None else reference
        )

        line = ax.plot(
            x,
            current_value.value,
            label=f"{label}$N={current_value.N}$ $({consistency:.1f}\%)$",
            alpha=current_value.N / y.N,
            linestyle=linestyle if i == (len(y) - 1) else ":",
        )

    if reference is None:
        plot_with_σ(
            x,
            y_final,
            label=f"{label}$N={y.N}$",
            linestyle=linestyle,
            ax=ax,
        )
    else:
        ax.fill_between(
            x,
            y_final.value - y_final.σ,
            y_final.value + y_final.σ,
            color=lighten_color(line[0].get_color(), 0.5),
        )

    return None


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def fancy_error(x, y, err, ax, **kwargs):
    line = ax.plot(
        x,
        y,
        **kwargs,
    )

    err = ax.fill_between(
        x,
        y + err,
        y - err,
        color=lighten_color(line[0].get_color(), 0.5),
        alpha=0.5,
    )

    return line, err


@wrap_plot
def plot_with_σ(
    x,
    y,
    ax=None,
    transform=lambda y: y,
    bath=None,
    strobe_frequency=None,
    strobe_tolerance=1e-3,
    hybrid=False,
    **kwargs,
):
    err = (y.σ[bath] if bath is not None else y.σ).real
    y_final = transform(y.value[bath] if bath is not None else y.value)

    strobe_mode = strobe_frequency is not None
    strobe_indices = None
    strobe_times = None
    strobe_style = dict(linestyle="none", marker="o", markersize=2) | kwargs

    if strobe_mode:
        strobe_times, strobe_indices = ut.strobe_times(
            x, strobe_frequency, strobe_tolerance
        )

        line = ax.errorbar(
            strobe_times,
            y_final[strobe_indices],
            err[strobe_indices],
            **strobe_style,
        )

        if not hybrid:
            return line

        kwargs["color"] = lighten_color(line[0].get_color(), 0.5)
        kwargs["label"] = ""

    return fancy_error(x, y_final, err, ax=ax, **kwargs)


@wrap_plot
def plot_diff_vs_sigma(
    x,
    y,
    reference,
    ax=None,
    label="",
    transform=lambda y: y,
    ecolor="yellow",
    ealpha=0.5,
    ylabel=None,
    bath=None,
):
    label = label + ", " if (len(label) > 0) else ""
    if bath is not None:
        y = y.for_bath(bath)
        reference = reference.for_bath(bath)

    ax.fill_between(
        x,
        0,
        reference.σ,
        color=ecolor,
        alpha=ealpha,
        label=rf"{label}$\sigma\, (N={reference.N})$",
    )

    for i in range(len(y)):
        current = y[i]

        not_last = current.N < y[-1].N
        consistency = current.consistency(reference)
        diff = abs(current - reference)

        ax.plot(
            x,
            diff.value,
            label=rf"{label}$N={current.N}$  $({consistency:.1f}\%)$",
            alpha=consistency / 100 if not_last else 1,
            linestyle=":" if not_last else "-",
            color=None if not_last else "red",
        )

    if ylabel is not None:
        if ylabel[0] == "$":
            ylabel = ylabel[1:-1]
        else:
            ylabel = rf"\text{{ {ylabel} }}"

        ax.set_ylabel(rf"$|{{{ylabel}}}_{{\mathrm{{ref}}}}-{{{ylabel}}}_{{N_i}}|$")


def plot_interaction_consistency(
    models,
    reference=None,
    label_fn=lambda model: f"$ω_c={model.ω_c:.2f}$",
    inset=None,
    bath=0,
    **kwargs,
):
    fig, (ax, ax2) = plt.subplots(ncols=2)

    ax3 = None
    if inset:
        slc, bounds = inset
        ax3 = ax.inset_axes(bounds)

    if reference:
        with aux.get_data(reference) as data:
            reference_energy = reference.interaction_energy(data, **kwargs)

    for model in models:
        with aux.get_data(model) as data:
            energy = model.interaction_energy(data, **kwargs).for_bath(bath)
            interaction_ref = model.interaction_energy_from_conservation(
                data, **kwargs
            ).for_bath(bath)

            self_consistency = energy.consistency(interaction_ref)
            normalizer = 1 / abs(energy.value).max()

            if reference:
                final_consistency = reference_energy.consistency(interaction_ref)

            _, _, (line, _) = plot_with_σ(
                data.time[:],
                energy,
                ax=ax,
            )

            plot_with_σ(
                data.time[:],
                interaction_ref,
                ax=ax,
                linestyle="--",
                color=lighten_color(line[0].get_color(), 0.8),
            )

            plot_with_σ(
                data.time[:],
                (energy - interaction_ref) * normalizer,
                label=label_fn(model)
                + fr", (${self_consistency:.0f}\%$"
                + (fr", ${final_consistency:.0f}\%$)" if reference else ")"),
                ax=ax2,
                color=line[0].get_color(),
            )

            if inset:
                plot_with_σ(
                    data.time[slc],
                    energy.slice(slc),
                    ax=ax3,
                    color=line[0].get_color(),
                )

                plot_with_σ(
                    data.time[slc],
                    interaction_ref.slice(slc),
                    ax=ax3,
                    linestyle="--",
                    color=lighten_color(line[0].get_color(), 0.8),
                )

    ax.set_xlabel(r"$\tau$")
    ax2.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\langle H_\mathrm{I}\rangle$")
    ax2.set_ylabel(
        r"$\langle (H_\mathrm{I}\rangle - \langle H_\mathrm{I}\rangle_\mathrm{ref}) / \langle H_\mathrm{I}\rangle_\mathrm{max}$"
    )
    ax2.legend()

    return fig, (ax, ax2, ax3)


def plot_interaction_consistency_development(
    models,
    reference=None,
    label_fn=lambda model: rf"$\omega_c={model.ω_c:.2f}$",
    **kwargs,
):
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)

    ax = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax2.set_xscale("log")
    ax2.set_yscale("log")

    ax3.set_xscale("log")
    ax3.set_yscale("log")

    if reference is not None:
        with aux.get_data(reference) as data:
            reference_energy = reference.interaction_energy(data, **kwargs)

    for model in models:
        with aux.get_data(model) as data:
            interaction_ref = model.interaction_energy_from_conservation(data, **kwargs)
            if reference is not None:
                energy = reference_energy
            else:
                energy = model.interaction_energy(data, **kwargs)

            normalizer = 1 / abs(energy.value).max()
            diff = abs(interaction_ref - energy) * normalizer

            ns, values, σs, diffs = [], [], [], []
            for N, val, σ in diff.aggregate_iterator:
                ns.append(N)
                values.append((val < σ).sum() / len(val[0]) * 100)
                σs.append(σ.mean() * normalizer)
                diffs.append(val.mean() * normalizer)

            σ_ref, σ_int = [], []

            for _, _, σ in interaction_ref.aggregate_iterator:
                σ_ref.append(σ.mean() * normalizer)

            for _, _, σ in energy.aggregate_iterator:
                σ_int.append(σ.mean() * normalizer)

            lines = ax.plot(
                ns,
                values,
                linestyle="--",
                marker=".",
                markersize=2,
                label=label_fn(model),
            )
            ax2.plot(
                ns,
                σs,
                label=label_fn(model),
                color=lines[0].get_color(),
            )

            ax2.plot(
                ns,
                diffs,
                label=label_fn(model),
                linestyle="--",
                color=lines[0].get_color(),
            )

            ax3.plot(
                ns,
                σ_ref,
                label=label_fn(model) + " (from conservation)",
                linestyle="--",
                color=lines[0].get_color(),
            )
            ax3.plot(ns, σ_int, label=label_fn(model) + " (direct)")

    ax.axhline(68, linestyle="-.", color="grey", alpha=0.5)
    ax.set_xlabel("$N$")
    ax.set_title("Consistency")
    ax2.set_xlabel("$N$")
    ax3.set_xlabel("$N$")
    ax2.set_ylabel(r"$[|\langle H_I\rangle|_\mathrm{max}]$")
    ax2.set_title(r"Mean Actual vs. Allowed deviation.")
    ax3.set_ylabel(r"$[|\langle H_I\rangle|_\mathrm{max}]$")
    ax3.set_title(r"Statistical Error of $\langle H_I\rangle$.")
    ax.set_ylabel(("" if reference else "Self-") + r"Consistency [$\%$]")
    ax.legend()

    straight_line = matplotlib.lines.Line2D(
        [], [], color="black", label=r"$\langle \sigma_\Delta\rangle$"
    )
    dashed_line = matplotlib.lines.Line2D(
        [], [], color="black", label=r"$\langle\Delta\rangle$", linestyle="--"
    )
    ax2.legend(handles=[straight_line, dashed_line])

    straight_line = matplotlib.lines.Line2D([], [], color="black", label=r"Direct")
    dashed_line = matplotlib.lines.Line2D(
        [], [], color="black", label=r"Energy Conservation", linestyle="--"
    )
    ax3.legend(handles=[dashed_line, straight_line])

    return fig, [ax, ax2, ax3]


def plot_consistency_development(value, reference, fig=None):
    if fig:
        (ax, ax2) = fig.subplots(nrows=2, sharex=True)
    else:
        fig, (ax, ax2) = (fig or plt).subplots(nrows=2, sharex=True)

    ax.set_xscale("log")
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    diff = abs(value - reference)

    ns, values, σs, max_diff = [], [], [], []

    for N, val, σ in diff.aggregate_iterator:
        ns.append(N)
        values.append(((val < σ).sum() / len(val[0])) * 100)
        σs.append(σ.mean())
        max_diff.append(val.mean())

    values = np.array(values)
    ns = np.array(ns)
    where_consistent = values >= 68

    values_consistent = values.copy()
    values_consistent[~where_consistent] = np.nan

    values_inconsistent = values.copy()
    values_inconsistent[where_consistent] = np.nan

    ax.plot(
        ns,
        values_consistent,
        linestyle="none",
        marker=".",
        markersize=5,
    )

    ax.plot(
        ns,
        values_inconsistent,
        linestyle="none",
        marker=".",
        markersize=5,
    )

    ax2.plot(ns, np.array(σs) / abs(reference).max(), label=r"$\langle \sigma\rangle$")
    ax2.plot(
        ns, np.array(max_diff) / abs(reference).max(), label=r"$\langle \Delta\rangle$"
    )

    ax.axhline(68, linestyle="-.", color="grey", alpha=0.5)
    ax2.set_xlabel("$N$")
    ax.set_ylabel(r"[$\%$]")
    ax.set_title(r"Consistency")
    ax2.set_ylabel(r"$[{J_\mathrm{max}}]$")
    ax2.set_title(r"Deviation")

    ax2.legend()

    return fig, [ax, ax2]


def plot_flow_bcf(models, label_fn=lambda model: f"$ω_c={model.ω_c:.2f}$", **kwargs):
    fig, ax = plt.subplots()
    for model in models:
        with aux.get_data(model) as data:
            flow = model.bath_energy_flow(data, **kwargs)
            _, _, (line, _) = plot_with_σ(
                data.time[:],
                flow,
                ax=ax,
                label=label_fn(model),
                bath=0,
                transform=lambda y: -y,
            )

            ax.plot(
                data.time[:],
                -model.L_expect * model.bcf_scale * model.bcf(data.time[:]).imag,
                linestyle="--",
                color=line[0].get_color(),
            )

    return fig, ax


def plot_energy_overview(
    model,
    ensemble_args=None,
    online=True,
    # system=True,
    # bath=True,
    # total=True,
    # flow=True,
    # interaction=True,
    bath_names=None,
    **kwargs,
):
    if not ensemble_args:
        ensemble_args = {}

    fig, ax = plt.subplots()
    ax.set_ylabel("Energy")
    ax.set_xlabel(r"$\tau$")

    data = None if online else aux.get_data(model)

    system_energy = model.system_energy(data, **ensemble_args)
    bath_energy = model.bath_energy(data, **ensemble_args)
    interaction_energy = model.interaction_energy(data, **ensemble_args)
    # flow = model.bath_energy_flow(data, **ensemble_args)

    plot_with_σ(model.t, system_energy, ax=ax, label="System", **kwargs)

    num_baths = interaction_energy.num_baths
    for bath in range(num_baths):
        label = bath_names[bath] if bath_names else bath + 1
        # plot_with_σ(
        #     model.t, flow, bath=bath, ax=ax, label=f"Flow {bath+1}", **kwargs
        # )
        plot_with_σ(
            model.t, bath_energy, bath=bath, ax=ax, label=f"Bath {label}", **kwargs
        )

        plot_with_σ(
            model.t,
            interaction_energy,
            ax=ax,
            bath=bath,
            label=f"Interaction {label}",
            **kwargs,
        )

    total = model.total_energy_from_power(data, **ensemble_args)
    plot_with_σ(
        model.t,
        total,
        ax=ax,
        label="Total",
        linestyle="--",
        **kwargs,
    )

    if data:
        data.close()

    return (fig, ax)


@wrap_plot
def plot_coherences(model, ax=None):
    with aux.get_data(model) as data:
        plot_with_σ(
            model.t,
            EnsembleValue(
                (
                    0,
                    np.abs(np.array(data.rho_t_accum.mean)[:, 0, 1]),
                    np.array(data.rho_t_accum.ensemble_std)[:, 0, 1],
                )
            ),
            ax=ax,
        )


@wrap_plot
def plot_distance_measures(model, strobe_indices, ax=None):
    with aux.get_data(model) as data:
        plot_with_σ(
            model.t, EnsembleValue(relative_entropy(data, strobe_indices[-1])), ax=ax
        )
        plot_with_σ(
            model.t, EnsembleValue(trace_distance(data, strobe_indices[-1])), ax=ax
        )


@wrap_plot
def plot_σ_development(ensemble_value, ax=None, **kwargs):
    # norm = abs(ensemble_value.value).max()
    ax.plot(
        ensemble_value.Ns,
        [σ.mean() for σ in ensemble_value.σs],
        marker=".",
        markersize=6,
        linestyle="dotted",
        **kwargs,
    )

    ax.set_ylabel("mean normalized $σ$")
    ax.set_xlabel("$N$")


def plot_multi_energy_overview(
    models,
    label_fn=lambda m: fr"$\omega_c={m.ω_c:.1f},\,\alpha(0)={m.bcf(0).real:.2f}$",
    ensemble_arg=dict(),
):
    fig, ((ax_sys, ax_flow), (ax_inter, ax_bath)) = plt.subplots(
        nrows=2, ncols=2, sharex=True
    )

    # flow_ins = ax_flow.inset_axes([0.4, 0.20, 0.5, 0.55])
    ax_sys.set_ylabel(r"$\langle H_\mathrm{S}\rangle$")
    ax_flow.set_ylabel("$J$")
    ax_inter.set_ylabel(r"$\langle H_\mathrm{I}\rangle$")
    ax_bath.set_ylabel(r"$\langle H_\mathrm{B}\rangle$")

    ax_sys.set_xlabel(r"$\tau$")
    ax_flow.set_xlabel(r"$\tau$")
    ax_inter.set_xlabel(r"$\tau$")
    ax_bath.set_xlabel(r"$\tau$")

    handles = []
    for model in models:
        with aux.get_data(model) as data:
            sys_energy = model.system_energy(data, **ensemble_arg)
            flow = model.bath_energy_flow(data, **ensemble_arg).for_bath(0)
            inter = model.interaction_energy(data, **ensemble_arg).for_bath(0)
            bath = model.bath_energy(data, **ensemble_arg).for_bath(0)

            _, _, (lines, _) = plot_with_σ(
                data.time[:],
                sys_energy,
                ax=ax_sys,
                label=label_fn(model),
            )
            handles.append(lines[0])

            plot_with_σ(
                data.time[:],
                flow,
                ax=ax_flow,
            )

            mask = np.logical_and(7 <= data.time[:], data.time[:] <= 9)
            # plot_with_σ(
            #     data.time[mask],
            #     flow.slice(mask),
            #     ax=flow_ins,
            # )

            plot_with_σ(
                data.time[:],
                inter,
                ax=ax_inter,
            )

            plot_with_σ(
                data.time[:],
                bath,
                ax=ax_bath,
            )

    ax_sys.legend(
        # ncol=2,
    )


@wrap_plot
def plot_ρ(model, i, j, ax=None, **kwargs):
    with aux.get_data(model) as data:
        return plot_with_σ(
            model.t,
            EnsembleValue(
                (data.rho_t_accum.mean[:, i, j], data.rho_t_accum.ensemble_std[:, i, j])
            ),
            ax=ax,
            **kwargs,
        )
