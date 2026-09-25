from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();all_rows=[]
 for path in sorted(a.input.glob("seed_*.json")):all_rows.extend(json.loads(path.read_text())["rows"])
 rows=[]
 for method in sorted({r["method"] for r in all_rows}):
  selected=[r for r in all_rows if r["method"]==method];mw=[r["oracle"]["total_renewable_mw"] for r in selected if r.get("oracle")];score=[r["oracle"]["f"] for r in selected if r.get("oracle") and r["oracle"]["f"] is not None];feasible=[r["oracle"]["converged"] and r["oracle"]["f"]<=1+1e-6 for r in selected if r.get("oracle") and r["oracle"]["f"] is not None];rows.append({"method":method,"seeds":len(selected),"total_mw_mean":float(np.mean(mw)) if mw else None,"total_mw_sd":float(np.std(mw,ddof=1)) if len(mw)>1 else 0.,"ac_score_mean":float(np.mean(score)) if score else None,"ac_feasible_fraction":float(np.mean(feasible)) if feasible else None})
 a.output.mkdir(parents=True,exist_ok=True);(a.output/"summary.json").write_text(json.dumps(rows,indent=2))
 with (a.output/"summary.csv").open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
if __name__=="__main__":main()
