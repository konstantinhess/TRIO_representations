"""Projection benchmark with the paper timing inclusion/exclusion rules."""
from __future__ import annotations
import argparse,json,time,hashlib,platform,importlib.metadata
from pathlib import Path
import cvxpy as cp
import numpy as np
from trio_paper.projection_2d import CachedEllipsoidProjector
from trio_paper.solvers.smooth_ipopt import baseline_smooth_mlp_ipopt_project
from trio_paper.solvers.smooth_scip import baseline_smooth_mlp_global_project
from trio_paper.solvers.cpwl_gurobi import cpwl_gurobi_project
from .evaluate import load_model
from .query_manifest import load_power_suite
from .ground_truth_projection import GroundTruthProjection,downstream_projection_quality

def json_default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,(np.integer,np.floating)):return value.item()
    raise TypeError(type(value).__name__)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class CachedICNNProjection:
    def __init__(self,model):
        d,w=model.input_dim,model.width;self.x=cp.Variable(d);z1,z2=cp.Variable(w),cp.Variable(w);self.query=cp.Parameter(d);self.threshold=cp.Parameter()
        soft=lambda value:cp.log_sum_exp(cp.vstack([np.zeros(w),value]),axis=0)
        fw,fb=model.input_first.weight.detach().numpy(),model.input_first.bias.detach().numpy();iw,ib=model.hidden_input.weight.detach().numpy(),model.hidden_input.bias.detach().numpy();ow=model.output_input.weight.detach().numpy()[0];ob=float(model.output_input.bias.detach())
        constraints=[z1>=soft(fw@self.x+fb),z2>=soft(model.hidden_nonnegative.detach().numpy()@z1+iw@self.x+ib+model.hidden_bias.detach().numpy()),model.output_nonnegative.detach().numpy()@z2+ow@self.x+ob<=self.threshold,cp.sum_squares(self.x)<=1]
        self.problem=cp.Problem(cp.Minimize(.5*cp.sum_squares(self.x-self.query)),constraints)
    def solve(self,query,threshold):
        self.query.value=query;self.threshold.value=threshold;value=self.problem.solve(solver="CLARABEL",tol_gap_abs=1e-9,tol_feas=1e-9,tol_gap_rel=1e-9,warm_start=False);success=self.problem.status in (cp.OPTIMAL,cp.OPTIMAL_INACCURATE) and self.x.value is not None
        return {"objective":float(value) if success else None,"point":np.asarray(self.x.value).tolist() if success else None,"status":self.problem.status,"globally_certified":bool(success and self.problem.status==cp.OPTIMAL)}

def make_reference(row):
    return GroundTruthProjection(tuple(row["z_star"]),row["projection_distance_star"],row["J_star"],row["raw_threshold"],row["reference_status"],row["branch"],row["branch_parameter"],row["dense_crosscheck_gap"])

def audit(result,query,row,power,cfg):
    return downstream_projection_quality(result.get("point"),query["x0"],query["threshold"],power,cfg["semantic_target"],make_reference(row),feasibility_tolerance=cfg["projection"]["true_feasibility_tolerance"])

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default=str(Path(__file__).with_name("config.json")));parser.add_argument("--runs",required=True);parser.add_argument("--manifest",required=True);parser.add_argument("--output",required=True);parser.add_argument("--methods",nargs="*",default=["trio","smooth_ipopt","smooth_scip","cpwl_gurobi","icnn_clarabel"]);args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    packages={name:importlib.metadata.version(name) for name in ("numpy","scipy","torch","cvxpy","clarabel")};(out/"environment.json").write_text(json.dumps({"python":platform.python_version(),"packages":packages,"config_sha256":sha(args.config),"manifest_sha256":sha(args.manifest)},indent=2))
    model_key={"trio":"trio","smooth_ipopt":"smooth","smooth_scip":"smooth","cpwl_gurobi":"cpwl","icnn_clarabel":"icnn"}
    for power in cfg["powers"]:
      queries,references,suite_hash=load_power_suite(Path(args.manifest),power)
      for seed in cfg["model_seeds"]:
        models={kind:load_model(args.runs,power,seed,kind,cfg) for kind in ("trio","smooth","cpwl","icnn")};rows=[];frozen=models["trio"].freeze();projector=CachedEllipsoidProjector(frozen);compiled={q["threshold"]:frozen.compile(q["threshold"]) for q in queries};icnn=CachedICNNProjection(models["icnn"])
        for method in args.methods:
          model=models[model_key[method]];results=[];timings=[]
          if method=="trio":
            def run_batch():return [{"point":None if point is None else point.tolist(),"objective":objective,"status":"optimal" if point is not None else "infeasible","globally_certified":point is not None} for query in queries for point,objective in [projector.project_compiled(compiled[query["threshold"]],query["x0"])]]
            run_batch();samples=[]
            for _ in range(cfg["projection"]["timing_repetitions"]):start=time.perf_counter();results=run_batch();samples.append((time.perf_counter()-start)/len(queries))
            timings=[float(np.mean(samples))]*len(queries)
          elif method=="icnn_clarabel":
            def run_batch():return [icnn.solve(query["x0"],query["threshold"]) for query in queries]
            run_batch();samples=[]
            for _ in range(cfg["projection"]["timing_repetitions"]):start=time.perf_counter();results=run_batch();samples.append((time.perf_counter()-start)/len(queries))
            timings=[float(np.mean(samples))]*len(queries)
          else:
            solver=(lambda query:baseline_smooth_mlp_ipopt_project(model,query["x0"],query["threshold"],cfg["projection"]["ipopt_starts"])) if method=="smooth_ipopt" else (lambda query:baseline_smooth_mlp_global_project(model,query["x0"],query["threshold"],cfg["projection"]["scip_timeout_seconds"])) if method=="smooth_scip" else (lambda query:cpwl_gurobi_project(model,query["x0"],query["threshold"],cfg["projection"]["scip_timeout_seconds"]))
            solver(queries[0])
            for query in queries:start=time.perf_counter();results.append(solver(query));timings.append(time.perf_counter()-start)
          for query,reference,result,elapsed in zip(queries,references,results,timings):rows.append({"power":power,"seed":seed,"method":method,"query_id":query["query_id"],"target_mass":query["target_mass"],"runtime_seconds":elapsed,**result,**audit(result,query,reference,power,cfg)})
        checkpoints={kind:sha(Path(args.runs)/f"p{power}"/f"seed_{seed}"/kind/"best.pt") for kind in models};(out/f"p{power}_seed{seed}.json").write_text(json.dumps({"query_suite_hash":suite_hash,"checkpoint_sha256":checkpoints,"timing_protocol":"prebuilt/cached; one warm-up; 20 batch repetitions for TRIO and CLARABEL; 60 individual solver timings otherwise","rows":rows},indent=2,default=json_default))
if __name__=="__main__":main()
