from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import torch
from trio_paper import build_trio, CPWLMLP, ICNN, parameter_count
from trio_paper.initialization import initialize_from_training
from trio_paper.training import train
from .data import load

def model_for(kind, q, config):
    if kind == "trio": return build_trio("wide_tanh", q, 2, 8)
    if kind == "cpwl": return CPWLMLP(2, tuple(config["cpwl_widths"][str(q)])).double()
    if kind == "icnn": return ICNN(2, int(config["icnn_widths"][str(q)])).double()
    raise ValueError(kind)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default=str(Path(__file__).with_name("config.json"))); parser.add_argument("--data",required=True); parser.add_argument("--output",required=True); parser.add_argument("--model",choices=["trio","cpwl","icnn"],required=True); parser.add_argument("--q",type=int,required=True); parser.add_argument("--seed",type=int,required=True); parser.add_argument("--smoke-steps",type=int)
    args=parser.parse_args(); config=json.loads(Path(args.config).read_text()); dataset=load(args.data); torch.manual_seed(args.seed); model=model_for(args.model,args.q,config)
    if args.model == "trio": initialize_from_training(model,dataset["train"]["normalized"],dataset["train"]["target"],"mixed_fps")
    protocol=dict(config["training"]); protocol["experiment_config_sha256"]=hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    if args.smoke_steps is not None: protocol.update(max_steps=args.smoke_steps,shield_steps=0,checkpoint_start_step=0,soft_steps=max(args.smoke_steps//2,1),patience_steps=args.smoke_steps+1,validation_every=max(args.smoke_steps//2,1))
    tensors=lambda x:torch.as_tensor(x,dtype=torch.float64)
    result=train(model,tensors(dataset["train"]["normalized"]),tensors(dataset["train"]["target"]),tensors(dataset["validation"]["normalized"]),tensors(dataset["validation"]["target"]),Path(args.output)/f"q{args.q}"/f"seed_{args.seed}"/args.model,protocol,trio=args.model=="trio",seed=args.seed)
    result["parameters"]=parameter_count(model); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
