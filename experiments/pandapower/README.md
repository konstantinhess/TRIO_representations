# Pandapower hosting capacity

This study uses the IEEE 30-bus case, five zero-reactive-power renewable
generators, and a 25 MW/site box. All predictors share one deterministic
20,000-point Sobol dataset. TRIO uses Q128 Wide-Tanh experts with 32 radial
units and mixed low-target FPS. Calibration is validation-only; solver
certificates concern the learned model, while physical feasibility is always
determined by a fresh AC power-flow evaluation.
