from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument("command",choices=["train","evaluate","manifest","projection","plot"]);p.add_argument("--config",default=str(Path(__file__).with_name("config.json")));p.add_argument("--output",default="results/complex_power");p.add_argument("--smoke",action="store_true");a=p.parse_args();cfg=json.loads(Path(a.config).read_text());root=Path(a.output)
 if a.command=="train":
  powers=cfg["powers"][:1] if a.smoke else cfg["powers"];seeds=cfg["model_seeds"][:1] if a.smoke else cfg["model_seeds"]
  for power in powers:
   for seed in seeds:
    for model in ("trio","smooth","cpwl","icnn"):
     cmd=[sys.executable,"-m","experiments.complex_power.train","--config",a.config,"--output",str(root/"runs"),"--power",str(power),"--seed",str(seed),"--model",model]
     if a.smoke:cmd += ["--smoke-steps","2"]
     subprocess.run(cmd,check=True)
 elif a.command=="evaluate":subprocess.run([sys.executable,"-m","experiments.complex_power.evaluate","--config",a.config,"--runs",str(root/"runs"),"--output",str(root/"evaluation"),*(["--resolution","31"] if a.smoke else [])],check=True)
 elif a.command=="manifest":
  if not (root/"query_manifest.json").exists():subprocess.run([sys.executable,"-m","experiments.complex_power.query_manifest","--config",a.config,"--output",str(root/"query_manifest.json")],check=True)
 elif a.command=="projection":subprocess.run([sys.executable,"-m","experiments.complex_power.projection","--config",a.config,"--runs",str(root/"runs"),"--manifest",str(root/"query_manifest.json"),"--output",str(root/"projection")],check=True)
 else:subprocess.run([sys.executable,"-m","experiments.complex_power.plot","--input",str(root/"projection"),"--output",str(root/"figures")],check=True)
if __name__=="__main__":main()
