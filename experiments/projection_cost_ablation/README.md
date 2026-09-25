# Projection-cost ablation

This untrained synthetic control holds a compiled union of 128 ellipsoids
fixed while increasing the complexity of the strictly monotone radial
backbone that is compiled away. It measures once-per-target compilation and
post-compilation exact projection separately. It is not an accuracy or
training experiment.

All variants use target `g=0.25`, the same centers, SPD matrices, desired
radii, 60 queries, and seeds 101–110. Geometry validation must pass before any
timer starts.

```bash
# Construction and geometry checks only
python -m experiments.projection_cost_ablation.run validate --seed 101

# One seed
python -m experiments.projection_cost_ablation.run seed --seed 101 --output results/projection_cost_ablation

# Restart-safe seeds 101–110
python -m experiments.projection_cost_ablation.run sweep --output results/projection_cost_ablation

# Mean ± sample SD table and figure
python -m experiments.projection_cost_ablation.aggregate --input results/projection_cost_ablation --output results/projection_cost_ablation/aggregate
```

Compilation uses 20 repetitions. Projection uses one warm-up followed by 20
alternating full-batch repetitions, normalized by 60 queries. Compilation,
radial inversion, cache/EVD construction, active-list construction, queries,
audits, and logging are outside the projection timer.
