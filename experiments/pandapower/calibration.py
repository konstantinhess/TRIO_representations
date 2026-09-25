from __future__ import annotations
import numpy as np,torch

def calibrate(model, validation_x, validation_y, config):
    with torch.inference_mode(): predictions=model(validation_x).detach().cpu().numpy()
    truth=validation_y.detach().cpu().numpy();infeasible=truth>1.0;c=config["calibration"]
    candidates=np.linspace(min(float(predictions.min()),float(c["minimum"])),float(c["maximum"]),int(c["candidate_count"]))
    rates=np.asarray([np.count_nonzero(infeasible & (predictions<=value))/max(np.count_nonzero(infeasible),1) for value in candidates])
    admissible=np.flatnonzero(rates<=float(c["false_feasible_rate_limit"]))
    if not len(admissible):raise RuntimeError("No validation threshold satisfies the false-feasible-rate limit")
    index=int(admissible[-1]);threshold=float(candidates[index]);feasible=truth<=1
    recall=float(np.count_nonzero(feasible&(predictions<=threshold))/max(np.count_nonzero(feasible),1))
    return {"threshold":threshold,"false_feasible_rate":float(rates[index]),"feasible_recall":recall,"candidate_count":len(candidates)}
