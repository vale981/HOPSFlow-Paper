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
import figsaver as fs
ray.shutdown()

#ray.init(address='auto')
ray.init()
from hops.util.logging_setup import logging_setup
import logging
logging_setup(logging.INFO)
import itertools
from hops.util.dynamic_matrix import ConstantMatrix

rot_models = []
weights = [.3, .6]
for weight in weights:
    off_ax = sc.make_model(0, 0)
    off_ax.H_bias = ConstantMatrix(.3 / 2 * qt.sigmax().full())
    rot_models.append(off_ax)

ot.integrate_online_multi(off_ax_models, 10, increment=10, analyze_kwargs=dict(every=10_000))
