import matplotlib
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pickle

from contextlib import contextmanager
import numpy as np

try:
    import tikzplotlib

except:
    tikzplotlib = None  # dirty

fig_path = Path(os.getcwd()) / "figures"
val_path = Path(os.getcwd()) / "values"


def export_fig(
    name, fig=None, x_scaling=1, y_scaling=0.4, tikz=True, save_pickle=True, **kwargs
):
    fig_path.mkdir(parents=True, exist_ok=True)
    if fig is None:
        fig = plt.gcf()

    w, _ = fig.get_size_inches()
    fig.set_size_inches((w * x_scaling, w * y_scaling))

    if not fig.get_constrained_layout():
        fig.tight_layout()

    fig.canvas.draw()

    # fig.savefig(fig_path / f"{name}.pdf")
    # fig.savefig(fig_path / f"{name}.svg")
    # fig.savefig(fig_path / f"{name}.pgf")
    fig.savefig(fig_path / f"{name}.pdf", bbox_inches="tight")

    if tikz and tikzplotlib:
        tikzplotlib.clean_figure()
        tikzplotlib.save(
            fig_path / f"{name}.tex",
            figure=fig,
            axis_width="\\figW",
            axis_height="\\figH",
            **kwargs,
        )

    if save_pickle:
        with open(fig_path / f"{name}.pickle", "wb") as file:
            pickle.dump(fig, file)


def scientific_round(val, *err, retprec=False):
    """Scientifically rounds the values to the given errors."""
    val, err = np.asarray(val), np.asarray(err)
    if len(err.shape) == 1:
        err = np.array([err])
        err = err.T
    err = err.T

    if err.size == 1 and val.size > 1:
        err = np.ones_like(val) * err

    if len(err.shape) == 0:
        err = np.array([err])

    if val.size == 1 and err.shape[0] > 1:
        val = np.ones_like(err) * val

    i = np.floor(np.log10(err))
    first_digit = (err // 10 ** i).astype(int)
    prec = (-i + np.ones_like(err) * (first_digit <= 3)).astype(int)
    prec = np.max(prec, axis=1)

    def smart_round(value, precision):
        value = np.round(value, precision)
        if precision <= 0:
            value = value.astype(int)
        return value

    if val.size > 1:
        rounded = np.empty_like(val)
        rounded_err = np.empty_like(err)
        for n, (value, error, precision) in enumerate(zip(val, err, prec)):
            rounded[n] = smart_round(value, precision)
            rounded_err[n] = smart_round(error, precision)

        if retprec:
            return rounded, rounded_err, prec
        else:
            return rounded, rounded_err

    else:
        prec = prec[0]
        if retprec:
            return (smart_round(val, prec), *smart_round(err, prec)[0], prec)
        else:
            return (smart_round(val, prec), *smart_round(err, prec)[0])


def tex_value(val, err=None, unit=None, prefix="", suffix="", prec=0, save=None):
    """Generates LaTeX output of a value with units and error."""

    if err:
        val, err, prec = scientific_round(val, err, retprec=True)
    else:
        val = np.round(val, prec)

    if prec == 0:
        val = int(val)
        if err:
            err = int(err)

    val_string = rf"{val:.{prec}f}" if prec > 0 else str(val)
    if err:
        val_string += rf"\pm {err:.{prec}f}" if prec > 0 else str(err)

    ret_string = r"\(" + prefix

    if unit is None:
        ret_string += val_string
    else:
        ret_string += rf"\SI{{{val_string}}}{{{unit}}}"

    ret_string += suffix + r"\)"

    if save is not None:
        val_path.mkdir(parents=True, exist_ok=True)

        with open(val_path / f"{save}.tex", "w") as f:
            f.write(ret_string)

    return ret_string


###############################################################################
#                                 SIDE EFFECTS                                #
###############################################################################

MPL_RC = {
    "lines.linewidth": 0.5,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.figsize": (3.4, 3.2),
    "text.usetex": True,
    "font.family": "serif",
    # "font.serif": ["Roman"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.constrained_layout.use": True,
    # "text.latex.preamble": r"\usepackage{mathtools}",
}

MPL_RC_POSTER = {
    "font.family": "sans",
    "text.usetex": False,
    "pgf.rcfonts": False,
    "lines.linewidth": 1.5,
    "font.size": 17.28,
    "axes.labelsize": 17.28,
    "axes.titlesize": 17.28,
    "legend.fontsize": 10,
    "xtick.labelsize": 14.4,
    "ytick.labelsize": 14.4,
}


@contextmanager
def hiro_style():
    with plt.style.context("seaborn-deep"):
        with matplotlib.rc_context(MPL_RC):
            yield True


plt.style.use("default")
plt.style.use("seaborn-paper")
matplotlib.rcParams.update(MPL_RC)
