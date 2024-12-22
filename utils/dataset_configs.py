import numpy as np
import pandas as pd
import torch

def ag_NP_function(X: torch.Tensor, df: pd.DataFrame):
    x1 = X[0].item()
    x2 = X[1].item()
    x3 = X[2].item()
    x4 = X[3].item()
    x5 = X[4].item()
    poss_resp = df.loc[(df["QAgNO3(%)"]==x1) & (df["Qpva(%)"]==x2) & (df["Qtsc(%)"]==x3) & (df["Qseed(%)"]==x4) & (df["Qtot(uL/min)"]==x5)]["loss"].values
    resp = np.random.choice(poss_resp, 1)[0]
    return resp

def autoAM_function(X: torch.Tensor, df: pd.DataFrame):
    x1 = X[0].item()
    x2 = X[1].item()
    x3 = X[2].item()
    x4 = X[3].item()
    poss_resp = df.loc[(df["Prime Delay"]==x1) & (df["Print Speed"]==x2) & (df["X Offset Correction"]==x3) & (df["Y Offset Correction"]==x4)]["Score"].values
    return poss_resp[0]

def crossed_barrel_function(X: torch.Tensor, df: pd.DataFrame):
    x1 = X[0].item()
    x2 = X[1].item()
    x3 = X[2].item()
    x4 = X[3].item()
    poss_resp = df.loc[(df["n"]==x1) & (df["theta"]==x2) & (df["r"]==x3) & (df["t"]==x4)]["toughness"].values
    resp = np.mean(poss_resp) + np.random.normal(loc=0, scale=2.5, size=1)[0]
    return resp
  

agnp = {"name": "AgNP",
        "path": "Benchmarking/datasets/AgNP_dataset.csv",
        "fct": ag_NP_function,
        "n_iters": 40}

auto_am = {"name": "AutoAM",
        "path": "Benchmarking/datasets/AutoAM_dataset.csv",
        "fct": autoAM_function,
        "n_iters": 30}

cb = {"name": "CrossedBarrel",
        "path": "Benchmarking/datasets/Crossed barrel_dataset.csv",
        "fct": crossed_barrel_function,
        "n_iters": 100}
        