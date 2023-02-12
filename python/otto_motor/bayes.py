from bayes_opt import BayesianOptimization
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
    k_max=3,
    bcf_terms=[4] * 2,
    truncation_scheme="simplex",
    driving_process_tolerances=[StocProcTolerances(1e-3, 1e-3)] * 2,
    thermal_process_tolerances=[StocProcTolerances(1e-3, 1e-3)] * 2,
    T=[0.5, 4],
    therm_methods=["tanhsinh", "tanhsinh"],
    Δ=1,
    num_cycles=3,
    Θ=1.5 / 0.05,
    dt=0.01,
    timings_H=p_H,
    timings_L=p_L,
    streaming_mode=True,
    shift_to_resonance=(False, True),
    L_shift=(0, 0),
)


def make_cycle(shift_c, shift_h):
    crazy_model = prototype.copy()
    (p_H, p_L) = timings(τ_mod, τ_I, 1)
    p_L = [list(timings) for timings in p_L]
    p_L[1][2] += τ_I
    p_L[1][3] += τ_I
    p_L[0][0] -= τ_I
    p_L[0][1] -= τ_I
    crazy_model.timings_H = p_H
    crazy_model.timings_L = tuple(tuple(timing) for timing in p_L)
    crazy_model.L_shift = (shift_c + τ_mod, shift_h)
    crazy_model.description = "Full Overlap with Shift"

    return crazy_model

def objective(shift_c, shift_h, N=500):
    print(shift_c, shift_h)
    model = make_cycle(shift_c, shift_h)
    ot.integrate_online(model, N)

    return -1 * model.power(steady_idx=-2).value

# Bounded region of parameter space
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
pbounds = {"shift_c": (-0.1, 0.5), "shift_h": (-0.1, 0.5)}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=1,
)
# load_logs(optimizer, logs=["./logs.json"]);

# logger = JSONLogger(path="./logs.json")
# optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
optimizer.probe(
    params={"shift_c": 0.15, "shift_h": 0.15},
    lazy=True,
)

optimizer.maximize(
    init_points=4,
    n_iter=100,
)

with aux.model_db(data_path=".data") as db:
    model = db["05a638feb440fd913b41a5be74fbdd5a6cc358f2b556e61e4005b8539ca15115"]["model_config"]
c=make_cycle(0.401813980810373, 0.302982197157591)
# aux.import_results(
#     other_data_path = "taurus/.data",
#     results_path = "./results",
#     other_results_path = "taurus/results",
#     interactive = False,
#     models_to_import = [model],
#     force = False,
# )
#ot.plot_cycle(c)
#model.L_shift
t, total = ot.val_relative_to_steady(model, model.total_energy_from_power(), steady_idx=-2)
pu.plot_with_σ(t, total)
model.power(steady_idx=-2)
