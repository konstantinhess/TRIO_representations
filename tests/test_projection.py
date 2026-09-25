import json,math
from pathlib import Path
import numpy as np
from experiments.complex_power.ground_truth_projection import project_raw_threshold,raw_field
from trio_paper.ellipsoid_optimization import minimize_linear_over_boxed_ellipsoid_union
from trio_paper.models import EllipsoidUnion

ROOT=Path(__file__).resolve().parents[1]

def test_true_projection_special_cases():
 q=np.array([.2,.3]);alpha=math.radians(25)
 assert project_raw_threshold(q,4,alpha,1.0).J_star==0
 assert project_raw_threshold(q,4,alpha,-1.1).z_star is None
 zero=project_raw_threshold(np.array([.7,0.]),4,alpha,0.0);assert zero.J_star>=0;assert abs(raw_field(zero.z_star,4,alpha))<1e-9
 end=project_raw_threshold(q,4,alpha,-1.0);assert abs(np.linalg.norm(end.z_star)-1)<1e-12

def test_boxed_ellipsoid_linear_optimization():
 certificate=EllipsoidUnion(np.array([[.5,.5]]),np.array([np.eye(2)]),np.array([True]),np.array([.04]),0.)
 optimum=minimize_linear_over_boxed_ellipsoid_union(certificate,-np.ones(2));assert np.allclose(optimum.point,[.5+math.sqrt(.02)]*2,atol=1e-8)
