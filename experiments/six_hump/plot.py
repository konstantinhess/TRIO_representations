from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument("--evaluation",required=True);p.add_argument("--output",required=True);a=p.parse_args();rows=list(csv.DictReader(open(Path(a.evaluation)/"per_threshold.csv")))
    capacities=sorted({int(r["q"]) for r in rows});seeds=sorted({int(r["seed"]) for r in rows});value={}
    for q in capacities:
      for seed in seeds:
       for model in ("cpwl","icnn","trio"):
        selected=[float(r["rmse"]) for r in rows if int(r["q"])==q and int(r["seed"])==seed and r["model"]==model];value[q,seed,model]=selected[0]
    fig,ax=plt.subplots(figsize=(6.2,4.2))
    x=[q*31 for q in capacities];icnn=[np.asarray([value[q,s,"icnn"]/value[q,s,"cpwl"] for s in seeds]) for q in capacities];trio=[np.asarray([value[q,s,"trio"]/value[q,s,"cpwl"] for s in seeds]) for q in capacities]
    ax.axhline(1,color="#222222",label="CPWL MLP reference");ax.errorbar(x,[v.mean() for v in icnn],yerr=[v.std(ddof=1) for v in icnn],fmt="o-",color="#174a7e",label="ICNN");ax.errorbar(x,[v.mean() for v in trio],yerr=[v.std(ddof=1) for v in trio],fmt="o-",color="#9b1c1c",label="TRIO (ours)")
    ax.set_xscale("log");ax.set_xlabel("Trainable parameters");ax.set_ylabel("RMSE ratio");ax.legend();fig.tight_layout();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out)
if __name__=="__main__":main()
