from pathlib import Path
import pandas as pd,torch

def load(path):
 frame=pd.read_csv(Path(path));features=[f"P_{i}_mw" for i in range(1,6)];out={}
 for split in ("train","validation","test"):
  rows=frame[frame.split==split];out[split]=(torch.tensor(rows[features].to_numpy()/25.,dtype=torch.float64),torch.tensor(rows.f.to_numpy(),dtype=torch.float64))
 return out
