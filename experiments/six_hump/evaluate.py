from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np, torch
from trio_paper.metrics import regression_metrics,mask_metrics,component_count
from .data import load,grid
from .dgp import six_hump_camel
from .train import model_for

def load_best(kind,q,seed,config,root):
    model=model_for(kind,q,config); payload=torch.load(Path(root)/f"q{q}"/f"seed_{seed}"/kind/"best.pt",map_location="cpu",weights_only=False); model.load_state_dict(payload["model_state_dict"]); return model.eval()

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default=str(Path(__file__).with_name("config.json")));p.add_argument("--data",required=True);p.add_argument("--runs",required=True);p.add_argument("--output",required=True);p.add_argument("--resolution",type=int);a=p.parse_args();cfg=json.loads(Path(a.config).read_text());ds=load(a.data);xx,yy,raw,norm=grid(cfg,a.resolution);truth_values=six_hump_camel(raw);rows=[]
    for q in cfg["capacities"]:
      for seed in cfg["model_seeds"]:
       for kind in ("trio","cpwl","icnn"):
        model=load_best(kind,q,seed,cfg,a.runs)
        with torch.inference_mode():test_pred=model(torch.as_tensor(ds["test"]["normalized"],dtype=torch.float64)).numpy();grid_pred=model(torch.as_tensor(norm,dtype=torch.float64)).numpy()
        base={"model":kind,"q":q,"seed":seed,**regression_metrics(ds["test"]["target"],test_pred)}
        for threshold in cfg["thresholds"]:
          predicted=grid_pred<=threshold;truth=truth_values<=threshold;m=mask_metrics(truth,predicted);rows.append({**base,"threshold":threshold,**m,"predicted_components":component_count(predicted.reshape(xx.shape),8),"true_components":component_count(truth.reshape(xx.shape),8)})
          if kind=="trio":
            compiled=model.freeze().compile(threshold); mismatch=np.count_nonzero(predicted!=compiled.membership(norm)); assert mismatch==0
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with (out/"per_threshold.csv").open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(f"wrote {len(rows)} rows")
if __name__=="__main__":main()
