from __future__ import annotations
import argparse,csv,json,pickle,re
from pathlib import Path
import numpy as np,torch
from trio_paper.baselines import CPWLMLP
from .data import grid
from .dgp import six_hump_camel

def cells(path):
 _,poly,boxes=pickle.loads(Path(path).read_bytes());array=lambda x:np.asarray(x.detach().cpu().numpy() if hasattr(x,"detach") else x,dtype=np.float64)
 return [(array(a).reshape(-1,2),array(b).reshape(-1),array(lo).reshape(-1),array(hi).reshape(-1)) for (a,b),(lo,hi) in zip(poly,boxes,strict=True)]
def membership(points,items):
 result=np.zeros(len(points),bool)
 for a,b,lo,hi in items:
  within=np.all((points>=lo-1e-9)&(points<=hi+1e-9),1);idx=np.flatnonzero(within);result[idx[np.all(points[idx]@a.T+b>=-1e-9,1)]]=True
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--run-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--budgets",nargs="+",type=int,required=True);p.add_argument("--threshold",type=float,default=.4);p.add_argument("--grid",type=int,default=1201);a=p.parse_args();cfg=json.loads(a.config.read_text());_,_,raw,norm=grid(cfg,a.grid);payload=torch.load(a.checkpoint,map_location="cpu",weights_only=False);state=payload["model_state_dict"];model=CPWLMLP(2,(state["linears.0.weight"].shape[0],state["linears.1.weight"].shape[0])).double();model.load_state_dict(state);values=model(torch.tensor(norm,dtype=torch.float64)).detach().numpy();learned=values<=a.threshold;truth=six_hump_camel(raw)<=a.threshold;rows=[]
 for budget in a.budgets:
  run=a.run_root/f"budget_{budget}"/"under";item=cells(run/"captured_representation/final_preimage.pkl");inner=membership(norm,item);bad=values[inner]-a.threshold;positive=bad[bad>1e-9];intersection=np.count_nonzero(inner&truth);union=np.count_nonzero(inner|truth);log=(a.run_root/f"budget_{budget}"/"premap2.log").read_text(encoding="utf-8",errors="replace");times=re.findall(r"Time cost:\s*([0-9.]+)",log);visited=re.findall(r"(\d+) branch and bound domains visited",log);rows.append({"budget":budget,"visited_boxes":int(visited[-1]) if visited else None,"cells":len(item),"construction_seconds":float(times[-1]) if times else None,"certified_coverage":float(inner.sum()/learned.sum()),"truth_iou":float(intersection/union),"inner_violations":len(positive),"max_violation":float(positive.max()) if len(positive) else 0.})
 a.output.mkdir(parents=True,exist_ok=True);(a.output/"audit.json").write_text(json.dumps(rows,indent=2));
 with (a.output/"audit.csv").open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
if __name__=="__main__":main()
