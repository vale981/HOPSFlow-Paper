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

model = OttoEngine(
      δ=[0.4, 0.4],
      ω_c=[1, 1],
      ψ_0=qt.basis([2], [1]),
      description=f"A basic near-markovian, weakly coupled Otto Cycle that actually works.",
      k_max=4,
      bcf_terms=[6] * 2,
      truncation_scheme="simplex",
      driving_process_tolerances=[StocProcTolerances(1e-4, 1e-4)] * 2,
      thermal_process_tolerances=[StocProcTolerances(1e-4, 1e-4)] * 2,
      T=[1, 10],
      therm_methods=["tanhsinh", "tanhsinh"],
      Δ=1,
      num_cycles=5,
      Θ=1.5 / 0.05,
      dt=0.001,
      timings_H=(0, 0.1, 0.5, 0.6),
      timings_L=((0.6, 0.7, 0.9, 1), (0.1, 0.2, 0.4, 0.5)),
      streaming_mode=True,
    shift_to_resonance=(False, False),
      # solver_args=dict(rtol=1e-3, atol=1e-3)
  )
ot.plot_cycle(model)

ot.plot_sd_overview(model)

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

model = OttoEngine(
    δ=[0.4, 0.4],
    ω_c=[2, 2],
    ψ_0=qt.basis([2], [1]),
    description=f"A basic near-markovian, weakly coupled Otto Cycle without the shift.",
    k_max=3,
    bcf_terms=[4] * 2,
    truncation_scheme="simplex",
    driving_process_tolerances=[StocProcTolerances(1e-4, 1e-4)] * 2,
    thermal_process_tolerances=[StocProcTolerances(1e-4, 1e-4)] * 2,
    T=[0, 2],
    therm_methods=["tanhsinh", "tanhsinh"],
    Δ=1,
    num_cycles=1,
    Θ=1.5 / 0.05,
    dt=0.001,
    timings_H=(0, 0.1, 0.5, 0.6),
    timings_L=((0.6, 0.7, 0.9, 1), (0.1, 0.2, 0.4, 0.5)),
    streaming_mode=True,
    shift_to_resonance=(False, False),
    #ω_s_extra=[.1, .1],
)

model_fft = model.copy()
model_fft.therm_methods = ["fft", "fft"]

ot.plot_cycle(model)

ot.plot_sd_overview(model)
