"""Two-logit bridge used by PREMAP2 in its separate Python environment."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn

class TwoLogitTarget(nn.Module):
    """Class zero is exactly the scalar event ``predictor(x) <= threshold``."""
    def __init__(self, predictor: nn.Module, threshold: float):
        super().__init__(); self.predictor=predictor; self.threshold=float(threshold)
    def forward(self,x):
        value=self.predictor(x).reshape(-1);return torch.stack((torch.full_like(value,self.threshold)-value,torch.zeros_like(value)),1)

def export_onnx(predictor,destination,threshold):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(TwoLogitTarget(predictor.float().eval(),threshold),torch.zeros((1,2),dtype=torch.float32),destination,input_names=["input"],output_names=["output"],opset_version=12,dynamic_axes={"input":{0:"batch"},"output":{0:"batch"}})

def write_vnnlib(destination):
    Path(destination).write_text("\n".join(["(declare-const X_0 Real)","(declare-const X_1 Real)","(declare-const Y_0 Real)","(declare-const Y_1 Real)","(assert (>= X_0 -1.0))","(assert (<= X_0 1.0))","(assert (>= X_1 -1.0))","(assert (<= X_1 1.0))","(assert (<= Y_0 Y_1))",""]))
