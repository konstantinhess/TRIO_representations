from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DISPLAY={"trio":"TRIO","smooth_ipopt":"Smooth MLP + IPOPT","smooth_scip":"Smooth MLP + SCIP","cpwl_gurobi":"CPWL MLP + Gurobi","icnn_clarabel":"ICNN + CLARABEL"}
COLORS={"trio":"#9b1c1c","smooth_ipopt":"#7a5195","smooth_scip":"#ef5675","cpwl_gurobi":"#2f4b7c","icnn_clarabel":"#00876c"}
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();rows=[]
 for path in a.input.glob("p*_seed*.json"):rows.extend(json.loads(path.read_text())["rows"])
 a.output.mkdir(parents=True,exist_ok=True);powers=sorted({r["power"] for r in rows});methods=list(DISPLAY)
 specifications=[("certification_true_feasible","Certified and true-feasible (%)",lambda rs:100*np.mean([bool(r.get("globally_certified")) and bool(r.get("true_feasible")) for r in rs])),("true_violation","Mean true violation",lambda rs:np.nanmean([r.get("true_violation",np.nan) for r in rs])),("objective_regret","Mean objective regret",lambda rs:np.nanmean([r.get("objective_regret",np.nan) if r.get("objective_regret") is not None else np.nan for r in rs])),("runtime","Mean runtime / query (s)",lambda rs:np.mean([r["runtime_seconds"] for r in rs]))]
 for filename,ylabel,metric in specifications:
  fig,ax=plt.subplots(figsize=(6.4,4.2))
  for method in methods:
   means=[];errors=[]
   for power in powers:
    per_seed=np.asarray([metric([r for r in rows if r["power"]==power and r["method"]==method and r["seed"]==seed]) for seed in sorted({r["seed"] for r in rows})],dtype=float);means.append(np.nanmean(per_seed));errors.append(np.nanstd(per_seed,ddof=1))
   if filename=="certification_true_feasible" and method=="smooth_ipopt":continue
   ax.errorbar(powers,means,yerr=errors,fmt="o-",label=DISPLAY[method],color=COLORS[method])
  ax.set(xlabel="Power p",ylabel=ylabel,xticks=powers);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(a.output/f"{filename}.pdf");plt.close(fig)
if __name__=="__main__":main()
