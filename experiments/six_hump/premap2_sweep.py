from __future__ import annotations
import argparse,os,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument("--locked-python",type=Path,required=True);p.add_argument("--premap2-root",type=Path,required=True);p.add_argument("--runs",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--seeds",nargs="+",type=int,default=range(101,111));p.add_argument("--budgets",nargs="+",type=int,default=[64,128,256,512,1024,2048]);a=p.parse_args();env={**os.environ,"PREMAP2_ROOT":str(a.premap2_root.resolve()),"TRIO_PREMAP_WORK":str(a.output.resolve()),"TRIO_PREMAP_OUTPUT":str(a.output.resolve())}
 for seed in sorted(a.seeds):
  checkpoint=a.runs/"q64"/f"seed_{seed}"/"cpwl"/"best.pt";adapter=a.output/f"seed_{seed}"/"adapter";subprocess.run([sys.executable,"-m","experiments.six_hump.premap2_prepare","--checkpoint",str(checkpoint),"--output",str(adapter)],check=True)
  for budget in sorted(a.budgets):
   tag=f"seed_{seed}/budget_{budget}";done=a.output/tag/"under/run_summary.json"
   if done.exists():continue
   log=a.output/f"seed_{seed}"/f"budget_{budget}"/"premap2.log";log.parent.mkdir(parents=True,exist_ok=True)
   with log.open("w",encoding="utf-8") as handle:subprocess.run([str(a.locked_python),str(Path(__file__).with_name("premap2_worker.py")),"--branch-budget",str(budget),"--tag",tag,"--onnx-path",str(adapter/"model.onnx"),"--vnnlib-path",str(adapter/"property.vnnlib")],check=True,env=env,stdout=handle,stderr=subprocess.STDOUT,text=True)
  subprocess.run([sys.executable,"-m","experiments.six_hump.premap2_audit","--config",str(Path(__file__).with_name("config.json")),"--checkpoint",str(checkpoint),"--run-root",str(a.output/f"seed_{seed}"),"--output",str(a.output/f"seed_{seed}"),"--budgets",*map(str,a.budgets)],check=True)
if __name__=="__main__":main()
