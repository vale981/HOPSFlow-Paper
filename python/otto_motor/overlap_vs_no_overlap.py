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
plt.rcParams['figure.figsize'] = (12,4)

def timings(τ_c, τ_i, percent_overlap=0):
    τ_cI = τ_c * (1-percent_overlap)

    τ_thI = (1 - 2 * τ_cI) / 2
    τ_th = (1 - 2 * τ_c) / 2
    τ_i_on = (τ_thI - 2*τ_i)
    timings_H = (0, τ_c, τ_c + τ_th, 2*τ_c + τ_th)

    timings_L_hot = (τ_cI, τ_cI + τ_i, τ_cI + τ_i + τ_i_on, τ_cI + 2 * τ_i + τ_i_on)

    timings_L_cold = tuple(time + timings_H[2] for time in timings_L_hot)

    return timings_H, (timings_L_cold, timings_L_hot)

τ_mod, τ_I = 0.15, 0.15
(p_H, p_L) = timings(τ_mod, τ_I, 0)
prototype = OttoEngine(
      δ=[0.4, 0.4],
      ω_c=[2, 2],
      ψ_0=qt.basis([2], [1]),
      description=f"Classic Cycle",
      k_max=4,
      bcf_terms=[6] * 2,
      truncation_scheme="simplex",
      driving_process_tolerances=[StocProcTolerances(1e-3, 1e-3)] * 2,
      thermal_process_tolerances=[StocProcTolerances(1e-3, 1e-3)] * 2,
      T=[.5, 4],
      therm_methods=["tanhsinh", "tanhsinh"],
      Δ=1,
      num_cycles=4,
      Θ=1.5 / 0.05,
      dt=0.001,
      timings_H=p_H,
      timings_L=p_L,
      streaming_mode=True,
      shift_to_resonance=(False, True),
      L_shift=(0, 0),
  )
ot.plot_cycle(prototype)

shifted_model = prototype.copy()
(p_H, p_L) = timings(τ_mod, τ_I, 1)
shifted_model.timings_H = p_H
shifted_model.timings_L = p_L
shifted_model.L_shift = (τ_mod, τ_mod)
shifted_model.description="Decoupling Overlap"
ot.plot_cycle(shifted_model)

left_shifted_model = prototype.copy()
(p_H, p_L) = timings(τ_mod, τ_I, 1)
left_shifted_model.timings_H = p_H
left_shifted_model.timings_L = p_L
left_shifted_model.L_shift = (0, 0)
left_shifted_model.description="Coupling Overlap"
ot.plot_cycle(left_shifted_model)

overlap_model = prototype.copy()
(p_H, p_L) =  timings(τ_mod, τ_I, 1)
p_L = [list(timings) for timings in p_L]
p_L[1][2] += τ_I
p_L[1][3] += τ_I

p_L[0][0] -= τ_I
p_L[0][1] -= τ_I
overlap_model.timings_H = p_H
overlap_model.timings_L = tuple(tuple(timing) for timing in p_L)
overlap_model.L_shift = (τ_mod, 0)
overlap_model.description="Full Overlap"
ot.plot_cycle(overlap_model)

crazy_model = prototype.copy()
(p_H, p_L) =  timings(τ_mod, τ_I, 1)
p_L = [list(timings) for timings in p_L]
p_L[1][2] += τ_I
p_L[1][3] += τ_I

p_L[0][0] -= τ_I
p_L[0][1] -= τ_I
crazy_model.timings_H = p_H
crazy_model.timings_L = tuple(tuple(timing) for timing in p_L)
crazy_model.L_shift = (τ_mod *2, τ_mod)
crazy_model.description="Full Overlap with Shift"
ot.plot_cycle(crazy_model)

less_crazy_model = shifted_model.copy()


less_crazy_model.L_shift = (τ_mod *2, τ_mod*2)
less_crazy_model.description="Large Shift without Overlap"
ot.plot_cycle(less_crazy_model)

optimized_crazy_model = crazy_model.copy()


optimized_crazy_model.L_shift = (τ_mod + 0.401813980810373, 0.302982197157591)
optimized_crazy_model.description="Large Shift without Overlap"
ot.plot_cycle(optimized_crazy_model)

models = [prototype, shifted_model, left_shifted_model, overlap_model, crazy_model, less_crazy_model]
