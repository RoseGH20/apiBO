from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
from typing import Dict, List
import torch


def get_prior_mean(all_dist_normed: np.ndarray, prior_cond: str, rep: int, dataset_name: str):
    if prior_cond == "good":
        if dataset_name =="AutoAM":
            idx = np.argsort(all_dist_normed)[:40:2][rep]
            dist = all_dist_normed[idx]
        else:
            idx = np.argsort(all_dist_normed)[rep]
            dist = all_dist_normed[idx]
    elif prior_cond == "misleading":
        idx = np.argsort(all_dist_normed)[::-1][rep]
        dist = all_dist_normed[idx]
    return idx, dist


def get_prior_std(X_test: torch.Tensor, prior_std_perc: float = 0.10):
    min_val, _ = X_test.min(dim=0)
    max_val, _ = X_test.max(dim=0)
    std = (max_val- min_val)*prior_std_perc
    std = std.numpy()
    return std


def create_prior(prior_mean: torch.Tensor, prior_std: np.ndarray, X_test: torch.Tensor):
    rv = multivariate_normal(mean=prior_mean.detach().numpy(), cov=prior_std)
    disc_prior = rv.pdf(X_test)
    return disc_prior


def get_grouped_df(df: pd.DataFrame, dataset_name: str):
    if dataset_name == "AgNP":
        grouped_df = df.groupby(["QAgNO3(%)", "Qpva(%)", "Qtsc(%)", "Qseed(%)", "Qtot(uL/min)"])["loss"].mean().reset_index()
        grouped_df.rename(columns={"loss": "gt_mean"}, inplace=True)
    elif dataset_name == "AutoAM":
        grouped_df = df.groupby(["Prime Delay", "Print Speed", "X Offset Correction", "Y Offset Correction"])["Score"].mean().reset_index()
        grouped_df.rename(columns={"Score": "gt_mean"}, inplace=True)
    elif dataset_name == "CrossedBarrel":
        grouped_df = df.groupby(["n", "theta", "r", "t"])["toughness"].mean().reset_index()
        grouped_df.rename(columns={"toughness": "gt_mean"}, inplace=True)
    return grouped_df


def check_config(config: Dict):
  valid_dataset_name = ["AgNP", "AutoAM", "CrossedBarrel"]
  valid_strategy = ["vbo", "pbi", "apibo", "pibo"]
  assert config["dataset"]["name"] in valid_dataset_name, "Invalid dataset name, please select a dataset from {}.".format(valid_dataset_name)

  for strat in config["strategy"]:
    strat in valid_strategy, "Invalid strategy name {}, please select a strategy from {}.".format(strat, valid_strategy)

  if "apibo" in config["strategy"]:
    assert "alpha" in config.keys(), "Please specify 'alpha' for apibo."

  if "pibo" in config["strategy"] and "beta" not in config.keys():
    config["beta"] = min(config["dataset"]["n_iters"] / 10, 5)
    print("A value was not specified for beta, setting it to min(n_iters/10, 5) = {}.".format(config["beta"]))

  if "pibo" in config["strategy"] or "apibo" in config["strategy"]:
    if "prior_std" not in config.keys():
      config["prior_std"] = 0.1
      print("A value was not specified for the standard deviation of the prior, setting it to 0.1.")

  if "nrnd" not in config.keys():
    config["nrnd"] = 3
    print("Using default number of initial random points, nrnd = 3.")

  if "nrep" not in config.keys():
    config["nrep"]=20
    print("Using default number of repetitions, nrep = 20.")


def vis_results(perf_explore: np.ndarray, config: Dict):

  x = np.arange(perf_explore.shape[-1])
  n_rep = config["nrep"]

  fig, ax = plt.subplots(1, 2, figsize = (12,4))

  for c in range(2):
    for s, strat in enumerate(config["strategy"]):
      if strat =="vbo":
        ax[c].plot(np.mean(perf_explore[s,0], axis =0), label = strat, color = "k")
        ax[c].fill_between(x, np.mean(perf_explore[s,0], axis = 0 ) + np.std(perf_explore[s,0], axis = 0 )/np.sqrt(n_rep), 
                           np.mean(perf_explore[s,0], axis = 0 ) - np.std(perf_explore[s,0], axis = 0 )/np.sqrt(n_rep), alpha = 0.1, color = "k")
      else:
        ax[c].plot(np.mean(perf_explore[s,c], axis =0), label = strat)
        ax[c].fill_between(x, np.mean(perf_explore[s,c], axis = 0 ) + np.std(perf_explore[s,c], axis = 0 )/np.sqrt(n_rep), 
                           np.mean(perf_explore[s,c], axis = 0 ) - np.std(perf_explore[s,c], axis = 0 )/np.sqrt(n_rep), alpha = 0.1)
    ax[c].set_xlabel("iterations")
  ax[0].set_ylabel("Performance (%)")
  ax[0].set_title("Good prior")
  ax[1].set_title("Misleading prior")

  plt.legend()
  plt.show()


def vis_paper_results(paths: List[str]):

  fig, ax = plt.subplots(1, 2, figsize = (12,4))
  labels = ["_vbo", "_pbi", "_apibo", "_pibo"]

  for c in range(2):
    for path in paths:

      for label in labels:
        if label in path:
            s_label = label
      perf_explore = np.load(path)["perf_explo"]

      if "AgNP" in path:
        perf_explore = (perf_explore*-1)+1
      x = np.arange(perf_explore.shape[2])
      n_rep = perf_explore.shape[1]

      if s_label =="_vbo":
        ax[c].plot(np.mean(perf_explore[0], axis =0), label = s_label[1:], color = "k")
        ax[c].fill_between(x, np.mean(perf_explore[0], axis = 0 ) + np.std(perf_explore[0], axis = 0 )/np.sqrt(n_rep), 
                           np.mean(perf_explore[0], axis = 0 ) - np.std(perf_explore[0], axis = 0 )/np.sqrt(n_rep), alpha = 0.1, color = "k")
      else:
        ax[c].plot(np.mean(perf_explore[c], axis =0), label = s_label[1:])
        ax[c].fill_between(x, np.mean(perf_explore[c], axis = 0 ) + np.std(perf_explore[c], axis = 0 )/np.sqrt(n_rep), 
                           np.mean(perf_explore[c], axis = 0 ) - np.std(perf_explore[c], axis = 0 )/np.sqrt(n_rep), alpha = 0.1)

    ax[c].set_xlabel("iterations")
  ax[0].set_ylabel("Performance (%)")
  ax[0].set_title("Good prior")
  ax[1].set_title("Misleading prior")

  plt.legend()
  plt.show()
  