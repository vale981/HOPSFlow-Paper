from timing_scan import *

ot.integrate_online_multi(models, 10_000, increment=2000, analyze_kwargs=dict(every=100))
