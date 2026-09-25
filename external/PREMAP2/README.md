# PREMAP2 setup

PREMAP2 is an external dependency and is intentionally not vendored here.

1. Create a separate Python **3.10.11** environment.
2. Clone `https://github.com/Aggrathon/Premap2` and checkout commit
   `cdb0f412f82ca7df3a7d72fcdf567a4a952c0e64`.
3. Install the exact packages in `requirements-lock.txt`. For the locked CPU
   Torch build use `pip install -r requirements-lock.txt --extra-index-url
   https://download.pytorch.org/whl/cpu`.
4. Apply `patches/two_logit_shape_fix.diff` and
   `patches/native_domain_serialization.diff` from the clone root.
5. Export the frozen CPWL MLP with
   `experiments/six_hump/premap2_adapter.py`.  It wraps the scalar predictor as
   `G(x)=[g-F(x),0]`; class zero is therefore exactly `F(x)<=g`.

The paper study uses input splitting and under-approximation only, at threshold
`g=0.4`, for budgets `64,128,256,512,1024,2048`.  Neuron splitting, outer
approximations, and the old scalar pilot path are outside the protocol.  The
PREMAP2 process runs in the separate locked environment; its certificate cells
are then audited on the independent 1201-by-1201 grid by the main environment.

The released parser uses a class-margin convention whose accepted VNNLIB syntax
is documented in the adapter.  Do not reverse the provided two-logit condition:
the apparently unusual sign is required by PREMAP2's internal class-margin path.
