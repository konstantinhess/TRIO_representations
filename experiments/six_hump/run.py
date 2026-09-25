from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
from .data import generate

def main():
    p=argparse.ArgumentParser();p.add_argument("command",choices=["prepare","train-all","evaluate"]);p.add_argument("--config",default=str(Path(__file__).with_name("config.json")));p.add_argument("--output",default="results/six_hump");p.add_argument("--smoke",action="store_true");a=p.parse_args();cfg=json.loads(Path(a.config).read_text());root=Path(a.output);data=root/"data.npz"
    if a.command=="prepare": generate(cfg,data);return
    if not data.exists():generate(cfg,data)
    if a.command=="train-all":
      seeds=cfg["model_seeds"][:1] if a.smoke else cfg["model_seeds"];capacities=cfg["capacities"][:1] if a.smoke else cfg["capacities"]
      for q in capacities:
       for seed in seeds:
        for model in ("trio","cpwl","icnn"):
         command=[sys.executable,"-m","experiments.six_hump.train","--config",a.config,"--data",str(data),"--output",str(root/"runs"),"--model",model,"--q",str(q),"--seed",str(seed)]
         if a.smoke:command += ["--smoke-steps","2"]
         subprocess.run(command,check=True)
    else: subprocess.run([sys.executable,"-m","experiments.six_hump.evaluate","--config",a.config,"--data",str(data),"--runs",str(root/"runs"),"--output",str(root/"evaluation"),*(["--resolution","31"] if a.smoke else [])],check=True)
if __name__=="__main__":main()
