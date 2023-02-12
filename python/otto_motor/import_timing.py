from timing_scan import *

aux.import_results(
    data_path="./.data",
    other_data_path="./taurus/.data_timing",
    results_path="./results",
    other_results_path="./taurus/results_timing",
    interactive=False,
    models_to_import=models,
    skip_checkpoints=False,
    force=True,
)

from timing_scan import *

aux.import_results(
    data_path="./.data",
    other_data_path="./taurus/.data_timing",
    results_path="./results",
    other_results_path="./taurus/results_timing",
    interactive=False,
    models_to_import=models[5:6],
    skip_checkpoints=False,
    force=True,
)
