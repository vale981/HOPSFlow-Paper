"""Random and unversioned utility functions."""
import numpy as np
from numpy.typing import NDArray
from statsmodels.nonparametric.smoothers_lowess import lowess


def strobe_times(time: NDArray[np.float64], frequency: float, tolerance: float = 1e-4):
    r"""
    Given a time array ``time`` and an angular frequency ``frequency`` (ω) the
    time points (and their indices) coinciding with :math:`2π / ω \cdot n` within the
    ``tolerance`` are being returned.
    """

    stroboscope_interval = 2 * np.pi / frequency
    sorted_times = np.sort(time)
    tolerance = min(tolerance, (sorted_times[1:] - sorted_times[:-1]).min() / 2)
    strobe_indices = np.where((time % stroboscope_interval) <= tolerance)[0]

    if len(strobe_indices) == 0:
        raise ValueError("Can't match the strobe interval to the times.")

    strobe_times = time[strobe_indices]

    return strobe_times, strobe_indices


def linspace_with_strobe(
    begin: float, end: float, N: int, strobe_frequency: float
) -> NDArray[np.float64]:
    """
    Like ``linspace`` but so that the time points defined by the
    stroboscope angular frequency ``strobe_frequency`` are included.
    """

    return np.unique(
        np.sort(
            np.concatenate(
                [
                    np.linspace(begin, end, N),
                    np.arange(begin, end, 2 * np.pi / strobe_frequency),
                ]
            )
        )
    )


def ergotropy(
    ρs: NDArray[np.complex128], H: NDArray[np.complex128]
) -> NDArray[np.float64]:
    """
    Calculates the ergotropy of the states ``ρs`` with respect to the
    hamiltonian ``H``.
    """

    energies = np.linalg.eigh(H)[0]

    pops = np.array([np.linalg.eigh(ρ)[0][::-1] for ρ in ρs])
    return (
        energies * np.diagonal(ρs, axis1=1, axis2=2).real[:, ::-1] - pops * energies
    ).real.sum(axis=1)


def smoothen(
    t: NDArray[np.float64], y: NDArray[np.float64], **kwargs
) -> NDArray[np.float64]:
    """
    Uses :any:`statsmodels.nonparametric.smoothers_lowess.lowess` to
    smoothen ``y`` indexed by ``t``.

    The ``kwargs`` are passed straight into
    :any:`statsmodels.nonparametric.smoothers_lowess.lowess`.
    """

    kwargs = dict(frac=0.1, it=0) | kwargs
    return lowess(y, t, is_sorted=True, return_sorted=False, **kwargs)
