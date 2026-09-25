from trio_paper import build_trio,SmoothMLP,CPWLMLP,ICNN,parameter_count

def build(kind,cfg):
 m=cfg["models"][kind]
 if kind=="trio":model=build_trio("wide_tanh",m["experts"],5,m["radial_units"])
 elif kind=="smooth":model=SmoothMLP(5,tuple(m["widths"])).double()
 elif kind=="cpwl":model=CPWLMLP(5,tuple(m["widths"])).double()
 elif kind=="icnn":model=ICNN(5,m["width"]).double()
 else:raise ValueError(kind)
 assert parameter_count(model)==m["parameters"],(kind,parameter_count(model),m["parameters"])
 return model
