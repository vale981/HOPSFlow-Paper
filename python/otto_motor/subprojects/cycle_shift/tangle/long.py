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

shifts = sc.make_shifts(extra_r=4)

long_models = [sc.make_model(shift, shift, switch_t=6.) for shift in shifts]
long_baseline = sc.make_model(0., 0., switch_t=6.)

ot.integrate_online_multi(long_models, 80_000, increment=10_000, analyze_kwargs=dict(every=10_000))
