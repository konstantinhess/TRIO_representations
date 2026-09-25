from __future__ import annotations
import argparse,copy,json
from dataclasses import asdict
from pathlib import Path
import numpy as np,pandas as pd
from .oracle import base_network,select_renewable_buses,RenewableOracle,sobol_box,assign_splits,freeze_network,normalization_metadata,write_json

def candidate_summary(net,buses,pmax,cfg,seed):
 bounds=np.column_stack((np.zeros(5),np.full(5,pmax)));oracle=RenewableOracle(copy.deepcopy(net),buses,bounds,cfg)
 zero=oracle.evaluate(np.zeros(5));maximum=oracle.evaluate(np.full(5,pmax));probe=[oracle.evaluate(x) for x in sobol_box(bounds,cfg["candidate_sobol_points"],seed)];path=[oracle.evaluate(np.full(5,t)) for t in np.linspace(0,pmax,cfg["candidate_path_points"])]
 return {"pmax_mw":pmax,"zero":asdict(zero),"all_maximum":asdict(maximum),"probe_feasible_fraction":float(np.mean([r.feasible for r in probe])),"probe_nonconvergent_fraction":float(np.mean([not r.converged for r in probe])),"equal_path_has_feasible_interior":any(r.feasible for r in path),"equal_path_ends_infeasible":not path[-1].feasible}

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default=str(Path(__file__).with_name("config.json")));p.add_argument("--output",required=True);p.add_argument("--smoke-points",type=int);a=p.parse_args();cfg=json.loads(Path(a.config).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True);net=base_network();buses=select_renewable_buses(net,cfg["renewables"]["count"]);assert buses==cfg["renewables"]["buses"],buses
 if a.smoke_points is None:
  candidates=[candidate_summary(net,buses,float(value),cfg,501+i) for i,value in enumerate(cfg["candidate_per_bus_pmax_mw"])];qualifying=[r for r in candidates if not r["all_maximum"]["feasible"] and r["equal_path_has_feasible_interior"] and r["equal_path_ends_infeasible"] and .10<=r["probe_feasible_fraction"]<=.90 and r["probe_nonconvergent_fraction"]<=.05]
  if not qualifying:raise RuntimeError("No candidate satisfies the committed nontriviality rule")
  pmax=float(min(qualifying,key=lambda r:r["pmax_mw"])["pmax_mw"]);assert pmax==float(cfg["selected_per_bus_pmax_mw"]);write_json(out/"candidate_bound_diagnostics.json",{"selection_rule":cfg["selection_rule"],"candidates":candidates})
 else:pmax=float(cfg["selected_per_bus_pmax_mw"])
 bounds=np.column_stack((np.zeros(5),np.full(5,pmax)));oracle=RenewableOracle(copy.deepcopy(net),buses,bounds,cfg);count=a.smoke_points or cfg["dataset"]["sobol_points"];points=sobol_box(bounds,count,cfg["dataset"]["seed"]);rows=[]
 for index,point in enumerate(points):
  row=asdict(oracle.evaluate(point));row.update({f"P_{j+1}_mw":float(v) for j,v in enumerate(point)});row["point_id"]=index;rows.append(row)
 frame=pd.DataFrame(rows)
 if a.smoke_points is None:
  if not bool(frame.converged.all()):raise RuntimeError("The deterministic 20,000-point draw contains nonconvergent samples")
  frame["split"]=assign_splits(len(frame),{"dataset":cfg["dataset"]});frame[["point_id","split"]].to_csv(out/"split_assignments.csv",index=False)
 else:frame["split"]="smoke"
 frame.to_csv(out/"hosting_capacity_dataset.csv",index=False);write_json(out/"normalization.json",normalization_metadata(bounds));write_json(out/"metadata.json",{"renewable_buses":buses,"bounds_mw":bounds.tolist(),"points":len(frame),"converged_fraction":float(frame.converged.mean()),"feasible_fraction":float(frame.feasible.mean())})
 if a.smoke_points is None:freeze_network(net,out)
if __name__=="__main__":main()
