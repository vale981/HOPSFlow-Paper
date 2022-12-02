model = OttoEngine(
    δ=[0.4, 0.4],
    ω_c=[1, 1],
    ψ_0=qt.basis([2], [1]),
    description=f"A basic near-markovian, weakly coupled Otto Cycle that actually works.",
    k_max=4,
    bcf_terms=[6] * 2,
    truncation_scheme="simplex",
    driving_process_tolerances=[StocProcTolerances(1e-4, 1e-4)] * 2,
    thermal_process_tolerances=[StocProcTolerances(1e-4, 1e-4)] * 2,
    T=[1, 10],
    therm_methods=["tanhsinh", "fft"],
    Δ=1,
    num_cycles=5,
    Θ=1.5 / 0.05,
    dt=0.001,
    timings_H=(0, 0.1, 0.5, 0.6),
    timings_L=((0.6, 0.7, 0.9, 1), (0.1, 0.2, 0.4, 0.5)),
    streaming_mode=True,
)

# model = OttoEngine(
#     δ=[0.4, 0.4],
#     ω_c=[1, 1],
#     ψ_0=qt.basis([2], [1]),
#     description=f"An otto cycle with longer cooling.",
#     k_max=3,
#     bcf_terms=[4] * 2,
#     truncation_scheme="simplex",
#     driving_process_tolerances=[StocProcTolerances(1e-3, 1e-3)] * 2,
#     thermal_process_tolerances=[StocProcTolerances(1e-3, 1e-3)] * 2,
#     T=[1, 10],
#     therm_methods=["tanhsinh", "fft"],
#     Δ=1,
#     num_cycles=4,
#     Θ=1.5 / 0.05,
#     dt=0.001,
#     timings_H=(0, 0.1, 0.3, 0.4),
#     timings_L=((0.4, 0.5, 0.9, 1), (0.1, 0.11, 0.29, 0.3)),
#     streaming_mode=True,
# )

ot.plot_cycle(model)

ot.plot_sd_overview(model)
