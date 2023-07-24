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
#plt.rcParams['figure.figsize'] = (12,4)
def make_model(Θ, δ):
    (p_H, p_L) = ot.timings(.06, .06)
    return OttoEngine(
          δ=[δ, δ],
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
          num_cycles=3 if δ >= .25 else 5,
          Θ=Θ,
          dt=0.001,
          timings_H=p_H,
          timings_L=p_L,
          streaming_mode=True,
          shift_to_resonance=(False, False),
          L_shift=(0, 0),
      )

weak_coupling_model = make_model(50, .2)
weak_coupling_model.num_cycles = 5

ot.integrate_online(weak_coupling_model, 50_000, analyze_kwargs=dict(every=10_000))
