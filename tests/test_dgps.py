import json,math
from pathlib import Path
import numpy as np
from experiments.six_hump.dgp import six_hump_camel,raw_to_normalized,normalized_to_raw
from experiments.six_hump.data import _split
from experiments.complex_power.dgp import raw_field,label,dataset

ROOT=Path(__file__).resolve().parents[1]

def test_six_hump_spots_and_determinism():
 cfg=json.loads((ROOT/"experiments/six_hump/config.json").read_text());minimum=np.array([[0.08984201368301331,-0.7126564032704135]])
 assert abs(float(six_hump_camel(minimum)[0])+1.031628453489877)<1e-12
 points=np.array([[-2.,-1.5],[0.,0.],[2.,1.5]]);assert np.allclose(normalized_to_raw(raw_to_normalized(points,cfg),cfg),points)
 assert all(np.array_equal(a,b) for a,b in zip(_split(8,2601,cfg),_split(8,2601,cfg)))

def test_complex_power_convention_and_splits():
 cfg=json.loads((ROOT/"experiments/complex_power/config.json").read_text());point=np.array([[1.,0.]])
 assert abs(raw_field(point,8)[0]-math.cos(math.radians(25)))<1e-15
 assert abs(label(point,8,cfg)[0]-(math.cos(math.radians(25))-cfg["projection_offset"]))<1e-15
 first=dataset(4,101,cfg);second=dataset(4,101,cfg);assert np.array_equal(first["test"][0],second["test"][0])
