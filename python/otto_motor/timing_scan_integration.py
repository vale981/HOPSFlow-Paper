from timing_scan import *

ot.integrate_online_multi(models, 1000, analyze_kwargs=dict(every=100))
ot.integrate_online_multi(models, 2000, analyze_kwargs=dict(every=100))
ot.integrate_online_multi(models, 4000, analyze_kwargs=dict(every=100))
