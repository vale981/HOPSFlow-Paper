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

import ray
ray.shutdown()
#ray.init(address='auto')
ray.init()

from hops.util.logging_setup import logging_setup
import logging
logging_setup(logging.INFO)
plt.rcParams['figure.figsize'] = (16,8)

model = OttoEngine(
    δ=[.4, .4],
    ω_c=[1, 1],
    ψ_0=qt.basis([2], [1]),
    description=f"An otto cycle on the way to finding the baseline.",
    k_max=4,
    bcf_terms=[6]*2,
    truncation_scheme="simplex",
    driving_process_tolerances=[StocProcTolerances(1e-3, 1e-3)] * 2,
    thermal_process_tolerances=[StocProcTolerances(1e-3, 1e-3)] * 2,
    T = [1, 10],
    therm_methods=["tanhsinh", "fft"],
    Δ=1,
    num_cycles=5,
    Θ=1/.05,
    dt=.001,
    timings_H=(0, .1, .5, .6),
    timings_L=((.6, .7, .9, 1), (.1, .2, .4, .5)),
    streaming_mode=True,
)

plt.plot(model.t, model.H.operator_norm(model.t) - 1, label="H")
plt.plot(model.t, model.coupling_operators[0].operator_norm(model.t) * 2, label="cold")
plt.plot(model.t, model.coupling_operators[1].operator_norm(model.t) * 2, label="hot")
plt.legend()

ω = np.linspace(.01, 3, 1000)
plt.plot(ω, model.full_thermal_spectral_density(0)(ω) * model.bcf_scales[0])
plt.plot(1, model.full_thermal_spectral_density(0)(1) * model.bcf_scales[0], marker="o")
plt.plot(ω, model.full_thermal_spectral_density(1)(ω) * model.bcf_scales[1])
plt.plot(2, model.full_thermal_spectral_density(1)(2) * model.bcf_scales[1], marker="o")
