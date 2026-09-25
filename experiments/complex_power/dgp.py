from __future__ import annotations
import math
import numpy as np
from trio_paper.sampling import uniform_unit_disk

def raw_field(points, power, rotation_degrees=25.0):
    points=np.asarray(points,dtype=np.float64);z=points[:,0]+1j*points[:,1];return np.real(np.exp(-1j*math.radians(rotation_degrees))*z**power)

def label(points,power,config):
    # Preserve the historical stored-label offset used by the accepted query manifest.
    return raw_field(points,power,config["semantic_target"]["rotation_degrees"])-float(config["projection_offset"])

def dataset(power,seed,config):
    sizes=config["dataset"]
    return {name:(lambda x:(x,label(x,power,config)))(uniform_unit_disk(sizes[f"{name}_size"],seed+delta)) for name,delta in (("train",0),("validation",1000),("test",2000))}
