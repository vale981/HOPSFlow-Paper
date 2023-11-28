import plot_utils as pu
from hiro_models.one_qubit_model import StocProcTolerances
from hiro_models.otto_cycle import OttoEngine, get_energy_gap
import hiro_models.model_auxiliary as aux
import numpy as np
import qutip as qt
import utilities as ut
import stocproc
import matplotlib.pyplot as plt
import otto_utilities as ot
import shift_cycle as sc
import ray
import figsaver as fs
ray.shutdown()

#ray.init(address='auto')
ray.init()
from hops.util.logging_setup import logging_setup
import logging
logging_setup(logging.INFO)

off_ax_models = []
weights = [.3, .6]
param_iter = lambda: itertools.product([3, 6], weights)
for switch_t, weight in param_iter():
    off_ax = sc.make_model(0, 0, switch_t=switch_t)
    off_ax.H_0  = 1 / 2 * (qt.sigmaz().full() + np.eye(2) + weight * qt.sigmax().full())
    # NOTE: the hamiltonians will be normalzed so that their smallest EV is 0 and the largest one is 1

    off_ax.H_1  = off_ax.H_0.copy()

    off_ax_models.append(off_ax)

ot.integrate_online_multi(off_ax_models, 10, increment=10, analyze_kwargs=dict(every=10_000))
