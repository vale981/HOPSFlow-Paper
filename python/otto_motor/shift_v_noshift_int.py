from overlap_vs_no_overlap import *
ot.integrate_online_multi(models[-1:], 100_000, increment=10_000, analyze_kwargs=dict(every=10_000))
