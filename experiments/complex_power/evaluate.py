from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
from trio_paper.metrics import regression_metrics,mask_metrics,component_count
from trio_paper.sampling import uniform_unit_disk
from .dgp import dataset,label
from .models import build

def load_model(root,power,seed,kind,config):
 model=build(kind,config);payload=torch.load(Path(root)/f"p{power}"/f"seed_{seed}"/kind/"best.pt",map_location="cpu",weights_only=False);model.load_state_dict(payload["model_state_dict"]);return model.eval()

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default=str(Path(__file__).with_name("config.json")));p.add_argument("--runs",required=True);p.add_argument("--output",required=True);p.add_argument("--resolution",type=int);a=p.parse_args();cfg=json.loads(Path(a.config).read_text());resolution=a.resolution or cfg["geometry"]["grid_resolution"]
 axis=np.linspace(-1,1,resolution);xx,yy=np.meshgrid(axis,axis,indexing="xy");grid=np.column_stack((xx.ravel(),yy.ravel()));disk=np.square(grid).sum(1)<=1;rows=[]
 for power in cfg["powers"]:
  for seed in cfg["model_seeds"]:
   data=dataset(power,seed,cfg);thresholds=np.quantile(data["train"][1],cfg["geometry"]["threshold_quantiles"]);truth_values=label(grid,power,cfg)
   for kind in ("trio","smooth","cpwl","icnn"):
    model=load_model(a.runs,power,seed,kind,cfg)
    with torch.inference_mode():test_pred=model(torch.as_tensor(data["test"][0],dtype=torch.float64)).numpy();grid_pred=model(torch.as_tensor(grid,dtype=torch.float64)).numpy()
    forward=regression_metrics(data["test"][1],test_pred)
    for threshold in thresholds:
     truth=(truth_values<=threshold)&disk;pred=(grid_pred<=threshold)&disk;m=mask_metrics(truth,pred);row={"power":power,"seed":seed,"model":kind,"threshold":threshold,**forward,**m,"true_components":component_count(truth.reshape(xx.shape),4),"predicted_components":component_count(pred.reshape(xx.shape),4)}
     if kind=="trio":row["compiled_mismatches"]=int(np.count_nonzero(pred!=(model.freeze().compile(threshold).membership(grid)&disk)))
     rows.append(row)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 with (out/"per_threshold.csv").open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=sorted({k for row in rows for k in row}));w.writeheader();w.writerows(rows)
if __name__=="__main__":main()
