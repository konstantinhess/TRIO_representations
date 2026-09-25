# Six-Hump Camel

The experiment learns the raw Six-Hump value on `[-2,2] x [-1.5,1.5]`, using
the exact normalization in `config.json`. TRIO uses Wide-Tanh radial laws and
training-only mixed FPS (half global, half within the lowest 10% of labels).
The 1201-by-1201 grid uses 8-connectivity.

PREMAP2 is an external, sound inner-set extraction procedure for the frozen
Q64 CPWL MLP. Its certified coverage is relative to that learned predictor;
truth IoU is measured separately against the analytic Six-Hump set.
