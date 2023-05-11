from hiro_models.one_qubit_model import StocProcTolerances
from hiro_models.otto_cycle import OttoEngine
import otto_utilities as ot
import qutip as qt
import numpy as np

T = 50


def make_model(shift_c, shift_h, switch_t=3.0, switch_t_sys=None, only_cold=False):
    switch_time = switch_t / T
    switch_time_sys = (switch_t_sys if switch_t_sys else switch_t) / T

    (p_H, p_L) = ot.timings(switch_time_sys, switch_time)
    return OttoEngine(
        δ=[0.7, 0.7],
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
        num_cycles=3,
        Θ=60,
        dt=0.001,
        timings_H=p_H,
        timings_L=p_L,
        streaming_mode=True,
        shift_to_resonance=(False, False),
        L_shift=(shift_c, 0 if only_cold else shift_h),
    )


def make_step(N=3, N_over=2):
    return 3.0 / (T * (N - N_over))


def make_shifts(N=3, N_over=2, extra_r=2):
    step = 3.0 / (T * (N - N_over))
    shifts = [round(shift * step, 3) for shift in range(-N, N + 1 + extra_r)]
    return np.array(shifts)
