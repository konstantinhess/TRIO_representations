from trio_paper import build_trio, SmoothMLP, CPWLMLP, ICNN, parameter_count

def build(kind,config):
    cfg=config["models"][kind]
    if kind=="trio":model=build_trio("broken_power",cfg["experts"],2)
    elif kind=="smooth":model=SmoothMLP(2,tuple(cfg["widths"])).double()
    elif kind=="cpwl":model=CPWLMLP(2,tuple(cfg["widths"])).double()
    elif kind=="icnn":model=ICNN(2,cfg["width"]).double()
    else:raise ValueError(kind)
    assert parameter_count(model)==cfg["parameters"],(kind,parameter_count(model),cfg["parameters"])
    return model
