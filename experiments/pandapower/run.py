from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument("command",choices=["prepare","train","optimize"]);p.add_argument("--config",default=str(Path(__file__).with_name("config.json")));p.add_argument("--output",default="results/pandapower");p.add_argument("--smoke",action="store_true");a=p.parse_args();cfg=json.loads(Path(a.config).read_text());root=Path(a.output)
 if a.command=="prepare":subprocess.run([sys.executable,"-m","experiments.pandapower.prepare","--config",a.config,"--output",str(root/"data"),*(["--smoke-points","8"] if a.smoke else [])],check=True);return
 if a.command=="train":
  seeds=cfg["model_seeds"][:1] if a.smoke else cfg["model_seeds"]
  for seed in seeds:
   for model in ("trio","smooth","cpwl","icnn"):
    cmd=[sys.executable,"-m","experiments.pandapower.train","--config",a.config,"--data",str(root/"data"/"hosting_capacity_dataset.csv"),"--output",str(root/"runs"),"--model",model,"--seed",str(seed)]
    if a.smoke:cmd += ["--smoke-steps","2"]
    subprocess.run(cmd,check=True)
 else:
  for seed in (cfg["model_seeds"][:1] if a.smoke else cfg["model_seeds"]):subprocess.run([sys.executable,"-m","experiments.pandapower.optimize","--config",a.config,"--data-root",str(root/"data"),"--runs",str(root/"runs"),"--output",str(root/"optimization"),"--seed",str(seed)],check=True)
if __name__=="__main__":main()
