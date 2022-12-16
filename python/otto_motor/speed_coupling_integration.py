from speed_coupling_scan import *

ot.integrate_online_multi(models, 10_000, increment=10, analyze_kwargs=dict(every=100))
