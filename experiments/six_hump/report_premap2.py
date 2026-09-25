from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from .data import grid
from .dgp import six_hump_camel
from .evaluate import load_best

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default=str(Path(__file__).with_name("config.json")));p.add_argument("--runs",required=True);p.add_argument("--audits",required=True);p.add_argument("--output",required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_text());_,_,raw,norm=grid(cfg);threshold=cfg["premap2"]["threshold"];truth=six_hump_camel(raw)<=threshold;rows=[]
 for seed in cfg["model_seeds"]:
  model=load_best("trio",64,seed,cfg,a.runs);frozen=model.freeze();start=time.perf_counter();certificate=frozen.compile(threshold);compile_seconds=time.perf_counter()-start;mask=certificate.membership(norm);rows.append({"method":"TRIO","seed":seed,"budget":"exact","time_seconds":compile_seconds,"certified_extraction_coverage":1.0,"truth_iou":np.count_nonzero(mask&truth)/np.count_nonzero(mask|truth),"representation_size":int(certificate.active.sum())})
  audit=json.loads((Path(a.audits)/f"seed_{seed}"/"audit.json").read_text())
  for item in audit:rows.append({"method":"PREMAP2","seed":seed,"budget":item["budget"],"time_seconds":item.get("construction_seconds"),"certified_extraction_coverage":item["certified_coverage"],"truth_iou":item["truth_iou"],"representation_size":item["cells"]})
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 with (out/"per_seed.csv").open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 (out/"per_seed.json").write_text(json.dumps(rows,indent=2))
if __name__=="__main__":main()
