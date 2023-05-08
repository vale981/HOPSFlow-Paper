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

proto = sc.make_model(0, 0)
off_ax_models = []
for weight in [.3, .6]:
    off_ax = proto.copy()
    off_ax.H_0  = 1 / 2 * (qt.sigmaz().full() + np.eye(2) + weight * qt.sigmax().full())
    off_ax.H_1  = off_ax.H_0.copy()

    off_ax_models.append(off_ax)

ot.integrate_online_multi(off_ax_models, 10_000, increment=10_000, analyze_kwargs=dict(every=10_000))
