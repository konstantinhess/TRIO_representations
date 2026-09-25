from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import numpy as np
from .data import grid
from .dgp import six_hump_camel
from .evaluate import load_best

def ellipse_patch(center,matrix,radius_squared,scale):
 raw_matrix=np.diag(scale)@matrix@np.diag(scale);values,vectors=np.linalg.eigh(raw_matrix);axes=2*np.sqrt(radius_squared/values);angle=np.degrees(np.arctan2(vectors[1,0],vectors[0,0]));raw_center=center/scale
 return Ellipse(raw_center,axes[0],axes[1],angle=angle,fill=False,color="#8b1a1a",lw=.45,alpha=.65)

def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default=str(Path(__file__).with_name("config.json")));p.add_argument("--runs",required=True);p.add_argument("--output",required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_text());xx,yy,raw,norm=grid(cfg);values=six_hump_camel(raw).reshape(xx.shape);scale=np.asarray(cfg["normalization"]["scale"])
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 for threshold in cfg["qualitative_thresholds"]:
  capacities=cfg["qualitative_capacities"];fig,axes=plt.subplots(1,len(capacities),figsize=(4.2*len(capacities),3.7),sharex=True,sharey=True)
  for ax,q in zip(axes,capacities):
   model=load_best("trio",q,cfg["qualitative_seed"],cfg,a.runs);certificate=model.freeze().compile(threshold);pred=certificate.membership(norm).reshape(xx.shape);truth=values<=threshold;inter=np.count_nonzero(pred&truth);union=np.count_nonzero(pred|truth);ax.contour(xx,yy,truth.astype(float),[.5],colors=["#00bcd4"],linewidths=1.5)
   for i in certificate.expert_indices:ax.add_patch(ellipse_patch(certificate.centers[i],certificate.matrices[i],certificate.squared_radii[i],scale))
   ax.set_title(f"Q={q}, active={certificate.active.sum()}, IoU={inter/union:.3f}");ax.set_aspect("equal")
  axes[-1].legend(handles=[Line2D([0],[0],color="#00bcd4",lw=1.5,label="True preimage"),Line2D([0],[0],color="#8b1a1a",lw=.7,label="TRIO")],loc="upper right",frameon=True)
  fig.tight_layout();fig.savefig(out/f"preimage_g_{str(threshold).replace('-','m').replace('.','p')}.pdf");plt.close(fig)
if __name__=="__main__":main()
