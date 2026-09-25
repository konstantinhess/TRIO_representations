from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from trio_paper.baselines import CPWLMLP
from .premap2_adapter import export_onnx,write_vnnlib

def main():
 p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--threshold",type=float,default=.4);a=p.parse_args();payload=torch.load(a.checkpoint,map_location="cpu",weights_only=False);state=payload["model_state_dict"];first=state["linears.0.weight"].shape[0];second=state["linears.1.weight"].shape[0];model=CPWLMLP(2,(first,second)).double();model.load_state_dict(state);a.output.mkdir(parents=True,exist_ok=True);export_onnx(model,a.output/"model.onnx",a.threshold);write_vnnlib(a.output/"property.vnnlib");(a.output/"metadata.json").write_text(json.dumps({"threshold":a.threshold,"widths":[first,second]},indent=2))
if __name__=="__main__":main()
