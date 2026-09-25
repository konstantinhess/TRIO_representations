from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np,torch
from trio_paper import build_trio,SmoothMLP,CPWLMLP,ICNN,parameter_count
from trio_paper.initialization import farthest_point_indices,mixed_low_target_fps_indices,initialize_from_training

ROOT=Path(__file__).resolve().parents[1]

def test_exact_parameter_counts():
    assert parameter_count(build_trio("broken_power",128,2))==1152
    assert parameter_count(SmoothMLP(2,(24,42)))==1165
    assert parameter_count(CPWLMLP(2,(24,42)))==1165
    assert parameter_count(ICNN(2,31))==1212
    assert parameter_count(build_trio("wide_tanh",16,2,8))==496
    assert parameter_count(build_trio("wide_tanh",128,5,32))==15104
    assert parameter_count(SmoothMLP(5,(117,121)))==15102
    assert parameter_count(CPWLMLP(5,(117,121)))==15102
    assert parameter_count(ICNN(5,116))==15086

def test_compilation_matches_live_models():
    torch.manual_seed(103);points=torch.randn(97,2,dtype=torch.float64)
    for model in (build_trio("broken_power",8,2),build_trio("wide_tanh",8,2,3)):
        threshold=.4;live=model(points).detach().numpy()<=threshold;compiled=model.freeze().compile(threshold).membership(points.numpy());assert np.array_equal(live,compiled)

def test_fixed_seed_and_initializers():
    torch.manual_seed(101);a=build_trio("wide_tanh",8,2,2);torch.manual_seed(101);b=build_trio("wide_tanh",8,2,2)
    assert all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values()))
    points=np.array([[0.,0.],[1.,0.],[-1.,0.],[0.,1.],[0.,-1.],[.5,.5],[-.5,-.5],[.2,-.8]])
    assert np.array_equal(farthest_point_indices(points,3),[0,1,2])
    rng=np.random.default_rng(9);large_points=rng.uniform(-1,1,(100,2));values=np.arange(100,dtype=float);rows=mixed_low_target_fps_indices(large_points,values,8,.1);initialize_from_training(a,large_points,values,"mixed_fps");assert np.allclose(a.centers.detach().numpy(),large_points[rows]);assert np.allclose(a.beta.detach().numpy(),values[rows])

def test_configs_use_expected_scientific_scope():
    complex_cfg=json.loads((ROOT/"experiments/complex_power/config.json").read_text());assert complex_cfg["powers"]==[4,6,8,10,12]
    six=json.loads((ROOT/"experiments/six_hump/config.json").read_text());assert six["premap2"]["budgets"]==[64,128,256,512,1024,2048]
