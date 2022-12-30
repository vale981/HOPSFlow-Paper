from timing_scan import *

ot.integrate_online_multi(models, 10_000, increment=2000, analyze_kwargs=dict(every=100), data_path=".data_timing", results_path="results_timing")

aux.import_results(
    data_path="./.data",
    other_data_path="./.data_timing",
    results_path="./results",
    other_results_path="./results_timing",
    interactive=False,
    models_to_import=models,
)
