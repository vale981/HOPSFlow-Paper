<<boilerplate>>

shifts = sc.make_shifts()
cold_models = [sc.make_model(shift, shift, switch_t=6., only_cold=True) for shift in shifts]

ot.integrate_online_multi(cold_models, 80_000, increment=10_000, analyze_kwargs=dict(every=10_000))
