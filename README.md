# TRIO paper experiments

This repository reproduces the paper experiments for **TRIO**, a framework for learning expressive forward predictors with exact preimage representations that support downstream inverse optimization.


## Installation

Use Python 3.12.4 and run from this directory:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install . --no-deps
```

Gurobi additionally requires a valid license. SCIP is used through PySCIPOpt,
IPOPT through CasADi, and CLARABEL through CVXPY. PREMAP2 uses a separate
Python 3.10 environment described in `external/PREMAP2/README.md`; it is not a
dependency of the main environment.

All commands below resolve inputs relative to this repository or accept an
explicit output directory. Generated files go beneath `results/` by default.

## Six-Hump Camel

```bash
python -m experiments.six_hump.run prepare --output results/six_hump
python -m experiments.six_hump.run train-all --output results/six_hump
python -m experiments.six_hump.run evaluate --output results/six_hump
python -m experiments.six_hump.plot --evaluation results/six_hump/evaluation --output results/six_hump/rmse_ratio.pdf
python -m experiments.six_hump.plot_qualitative --runs results/six_hump/runs --output results/six_hump/figures
```

The configuration contains the qualitative Q=16/64/256 panels at thresholds
-0.1 and 0.9, the ten-seed paired capacity ratios at Q=16/32/64/128/256, and
the Q64 threshold-0.4 PREMAP2 comparison at budgets 64--2048. PREMAP2 export
uses `experiments/six_hump/premap2_adapter.py`.

After the separate environment is installed, launch the restart-safe PREMAP2
sweep and audit each seed with:

```bash
python -m experiments.six_hump.premap2_sweep --locked-python <py310-python> --premap2-root <Premap2-clone> --runs results/six_hump/runs --output results/six_hump/premap2
python -m experiments.six_hump.report_premap2 --runs results/six_hump/runs --audits results/six_hump/premap2 --output results/six_hump/premap2_table
```

## Complex powers

```bash
python -m experiments.complex_power.run train --output results/complex_power
python -m experiments.complex_power.run evaluate --output results/complex_power
python -m experiments.complex_power.aggregate_forward --input results/complex_power/evaluation --output results/complex_power/forward_summary
python -m experiments.complex_power.run manifest --output results/complex_power
python -m experiments.complex_power.run projection --output results/complex_power
python -m experiments.complex_power.run plot --output results/complex_power
```

The default reported powers are p=4,6,8,10,12. Forward geometry uses 19
training-label quantiles and an independent 801-by-801 disk grid. Projection
uses three equal-mass targets and 20 deterministic infeasible queries per
target. The analytic true-DGP reference is computed outside all timed solver
regions. SCIP has a 30-second query limit; IPOPT uses eight deterministic
starts. Continuous summaries use mean and sample SD, preserving missing values.

## Pandapower renewable hosting capacity

```bash
python -m experiments.pandapower.run prepare --output results/pandapower
python -m experiments.pandapower.run train --output results/pandapower
python -m experiments.pandapower.run optimize --output results/pandapower
python -m experiments.pandapower.aggregate --input results/pandapower/optimization --output results/pandapower/table
```

Preparation reconstructs `pandapower.networks.case30`, verifies the graph-FPS
renewable buses `[7,17,25,2,15]`, and regenerates the deterministic 20,000-point
Sobol dataset. Calibration uses validation data only: among 50,001 candidates,
it selects the largest threshold whose validation false-feasible rate is at
most 0.001. Every returned design is re-evaluated by the AC oracle.

## Layout

- `src/trio_paper/`: shared TRIO, baselines, compilation, projection, timing,
  and optional-solver adapters.
- `experiments/`: one compact package per scientific experiment.
- `external/PREMAP2/`: setup instructions, lock file, and compatibility patches.
- `tests/`: focused numerical, structural, and portability checks.
- `results/`: initially empty generated-output root.

Training sweeps, SCIP projection batches, and the 20,000-point AC dataset are
the runtime-heavy components. Training runners skip completed `metrics.json`
runs, making completed-run restarts safe; they do not claim bitwise mid-run
optimizer-state resumption.
