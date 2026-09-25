import json
from pathlib import Path
import numpy as np,torch
from experiments.pandapower.oracle import base_network,select_renewable_buses,RenewableOracle
from experiments.pandapower.calibration import calibrate

ROOT=Path(__file__).resolve().parents[1]

def test_oracle_smoke():
 cfg=json.loads((ROOT/"experiments/pandapower/config.json").read_text());net=base_network();buses=select_renewable_buses(net,5);assert buses==[7,17,25,2,15]
 result=RenewableOracle(net,buses,np.column_stack((np.zeros(5),np.full(5,25.))),cfg).evaluate(np.full(5,10.));assert result.converged and np.isfinite(result.f)

def test_validation_only_calibration():
 class Identity(torch.nn.Module):
  def forward(self,x):return x[:,0]
 cfg={"calibration":{"candidate_count":101,"minimum":.8,"maximum":1.,"false_feasible_rate_limit":.001}}
 x=torch.tensor([[.5],[.9],[1.1],[1.2]],dtype=torch.float64);y=torch.tensor([.8,.9,1.1,1.2],dtype=torch.float64);out=calibrate(Identity(),x,y,cfg);assert out["false_feasible_rate"]==0 and out["threshold"]<1.1
