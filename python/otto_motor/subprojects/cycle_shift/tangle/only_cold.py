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
import shift_cycle as sc
import ray
ray.shutdown()

#ray.init(address='auto')
ray.init()
from hops.util.logging_setup import logging_setup
import logging
logging_setup(logging.INFO)
plt.rcParams['figure.figsize'] = (12,4)

shifts = sc.make_shifts()
cold_models = [sc.make_model(shift, shift, switch_t=6., only_cold=True) for shift in shifts]

ot.integrate_online_multi(cold_models, 80_000, increment=10_000, analyze_kwargs=dict(every=10_000))
