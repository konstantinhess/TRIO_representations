from __future__ import annotations
import argparse,copy,json,subprocess,sys,tempfile,time,hashlib,platform,importlib.metadata
from pathlib import Path
import numpy as np,pandapower as pp,torch
from trio_paper.ellipsoid_optimization import minimize_linear_over_boxed_ellipsoid_union
from .calibration import calibrate
from .data import load
from .hosting_solvers import build_relu_gurobi_hosting_model,solve_gurobi_hosting_model,solve_tanh_ipopt_hosting,solve_icnn_clarabel_hosting
from .models import build
from .oracle import RenewableOracle

def json_default(value):
 if isinstance(value,np.ndarray):return value.tolist()
 if isinstance(value,(np.integer,np.floating)):return value.item()
 raise TypeError(type(value).__name__)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_best(root,seed,kind,cfg):
 model=build(kind,cfg);payload=torch.load(Path(root)/f"seed_{seed}"/kind/"best.pt",map_location="cpu",weights_only=False);model.load_state_dict(payload["model_state_dict"]);return model.eval()

def solve_scip(model,threshold,timeout):
 layers=[{"weight":layer.weight.detach().numpy().tolist(),"bias":layer.bias.detach().numpy().tolist()} for layer in model.network if hasattr(layer,"weight")]
 request={"layers":layers,"input_dim":5,"threshold":threshold,"timeout_seconds":timeout,"lower":0.,"upper":1.,"objective_weights":[1.]*5}
 with tempfile.TemporaryDirectory(prefix="trio_hosting_scip_") as folder:
  source,destination=Path(folder)/"request.json",Path(folder)/"result.json";source.write_text(json.dumps(request));subprocess.run([sys.executable,str(Path(__file__).with_name("scip_worker.py")),str(source),str(destination)],check=True,timeout=timeout+30);return json.loads(destination.read_text())

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default=str(Path(__file__).with_name("config.json")));p.add_argument("--data-root",required=True);p.add_argument("--runs",required=True);p.add_argument("--output",required=True);p.add_argument("--seed",type=int,required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_text());data=load(Path(a.data_root)/"hosting_capacity_dataset.csv");net=pp.from_pickle(str(Path(a.data_root)/"network"/"case30_frozen.p"));oracle=RenewableOracle(copy.deepcopy(net),cfg["renewables"]["buses"],np.column_stack((np.zeros(5),np.full(5,cfg["selected_per_bus_pmax_mw"]))),cfg);rows=[]
 for kind,method in (("trio","TRIO"),("smooth","Smooth MLP + IPOPT"),("smooth","Smooth MLP + SCIP"),("cpwl","CPWL MLP + Gurobi"),("icnn","ICNN + CLARABEL")):
  model=load_best(a.runs,a.seed,kind,cfg);cal=calibrate(model,*data["validation"],cfg);threshold=cal["threshold"]
  started=time.perf_counter()
  if method=="TRIO":
   compile_started=time.perf_counter();certificate=model.freeze().compile(threshold);compile_seconds=time.perf_counter()-compile_started;solve_started=time.perf_counter();opt=minimize_linear_over_boxed_ellipsoid_union(certificate,-np.ones(5));solve_seconds=time.perf_counter()-solve_started;result={"point_normalized":opt.point.tolist(),"globally_certified":True,"status":"optimal","active_experts":int(certificate.active.sum()),"compile_seconds":compile_seconds,"solve_seconds":solve_seconds}
  elif method.endswith("IPOPT"):result=solve_tanh_ipopt_hosting(model,threshold,cfg["optimization"]["ipopt_starts"],cfg["optimization"]["ipopt_rng_seed"])
  elif method.endswith("SCIP"):result=solve_scip(model,threshold,cfg["optimization"]["scip_timeout_seconds"])
  elif method.endswith("Gurobi"):
   problem,inputs,_=build_relu_gurobi_hosting_model(model,threshold);result=solve_gurobi_hosting_model(problem,inputs)
  else:result=solve_icnn_clarabel_hosting(model,threshold)
  elapsed=time.perf_counter()-started;point=result.get("point_normalized") or result.get("point")
  oracle_result=None if point is None else oracle.evaluate(25*np.asarray(point)).__dict__
  rows.append({"method":method,"seed":a.seed,"calibration":cal,"solver":result,"wall_seconds":elapsed,"oracle":oracle_result})
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);checkpoints={kind:sha(Path(a.runs)/f"seed_{a.seed}"/kind/"best.pt") for kind in ("trio","smooth","cpwl","icnn")};metadata={"python":platform.python_version(),"packages":{name:importlib.metadata.version(name) for name in ("numpy","torch","pandapower","cvxpy","clarabel")},"config_sha256":sha(a.config),"checkpoint_sha256":checkpoints};(out/f"seed_{a.seed}.json").write_text(json.dumps({"metadata":metadata,"rows":rows},indent=2,default=json_default))
if __name__=="__main__":main()
