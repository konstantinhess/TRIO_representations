import torch
from trio_paper.baselines import CPWLMLP
from experiments.six_hump.premap2_adapter import TwoLogitTarget

def test_two_logit_equivalence():
 torch.manual_seed(1);model=CPWLMLP(2,(4,5)).double();points=torch.randn(50,2,dtype=torch.float64);threshold=.4;logits=TwoLogitTarget(model,threshold)(points)
 assert torch.equal(logits[:,0]>=logits[:,1],model(points)<=threshold)
