#!/usr/bin/env python
# fit_subject_hssm_reg.py — single-subject, non-GPU (CPU/JAX)

import os
import argparse
import numpy as np
import pandas as pd
import arviz as az
from itertools import combinations

import hssm
from functools import partial
from hssm.utils import decorate_atomic_simulator
from hssm.distribution_utils.dist import make_hssm_rv
from ssms.basic_simulators.simulator import simulator
from hssm.likelihoods.rldm import make_rldm_logp_op

# (optional) tiny JAX cache on CPU
# try:
#     from jax.experimental import compilation_cache
#     compilation_cache.compilation_cache.set_cache_dir("/tmp/jax_cache")
# except Exception:
#     pass

def add_ushared_udiff(dataset, subj_col="participant_id", alpha0=1.0, beta0=1.0):
    df = dataset.copy()
    n = int(df["state2"].max()) + 1
    pair_list = np.array(list(combinations(range(n), 2)), dtype=np.int64)

    def _run_pair(g):
        pair_idx = g["state1"].to_numpy(np.int64)
        resp = g["response"].to_numpy(np.int64)
        s2 = g["state2"].to_numpy(np.int64)
        trial_pairs = pair_list[pair_idx]
        rows = np.arange(len(g))
        chosen_state = trial_pairs[rows, resp]
        is_common = (s2 == chosen_state).astype(np.int64)
        K = len(pair_list); alpha = np.full(K, alpha0); beta = np.full(K, beta0)
        var_ts = np.empty(len(g))
        for t, k in enumerate(pair_idx):
            a, b = alpha[k], beta[k]
            var_ts[t] = (a*b)/((a+b)**2*(a+b+1.0))
            if is_common[t]: alpha[k] = a+1.0
            else:            beta[k]  = b+1.0
        return var_ts

    def _run_state(g):
        pair_idx = g["state1"].to_numpy(np.int64)
        resp = g["response"].to_numpy(np.int64)
        s2 = g["state2"].to_numpy(np.int64)
        trial_pairs = pair_list[pair_idx]
        rows = np.arange(len(g))
        chosen_state = trial_pairs[rows, resp]
        is_common = (s2 == chosen_state).astype(np.int64)
        alpha = np.full(n, alpha0); beta = np.full(n, beta0)
        var_ts = np.empty(len(g))
        for t, k in enumerate(chosen_state):
            a, b = alpha[k], beta[k]
            var_ts[t] = (a*b)/((a+b)**2*(a+b+1.0))
            if is_common[t]: alpha[k] = a+1.0
            else:            beta[k]  = b+1.0
        return var_ts

    df["uncertainty_pair"] = np.nan
    df["uncertainty_state"] = np.nan
    for _, g in df.groupby(subj_col, sort=False):
        df.loc[g.index, "uncertainty_pair"] = _run_pair(g)
        df.loc[g.index, "uncertainty_state"] = _run_state(g)

    z = df.groupby(subj_col)[["uncertainty_pair","uncertainty_state"]].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8)
    )
    df["U_pair_z"] = z["uncertainty_pair"]; df["U_state_z"] = z["uncertainty_state"]
    root2 = np.sqrt(2.0)
    df["U_shared"] = (df["U_pair_z"] + df["U_state_z"]) / root2
    df["U_diff"]   = (df["U_pair_z"] - df["U_state_z"]) / root2
    return df

def create_dummy_simulator():
    def sim_wrapper(simulator_fun, theta, model, n_samples, random_state, **kwargs):
        return np.column_stack([np.random.uniform(0.2,0.6,n_samples),
                                np.random.randint(0,2,n_samples)])
    wrapped = partial(simulator, model="custom", n_samples=1)
    return decorate_atomic_simulator(model_name="custom", choices=[0,1], obs_dim=2)(wrapped)

def load_single_subject(csvfile, subj_id):
    df = pd.read_csv(csvfile, index_col=0).rename(columns={
        "subj_idx":"participant_id","trial":"trial_id","response1":"response","rt1":"rt"
    })
    for c in ["participant_id","trial_id","response","response2","state1","state2"]:
        if c in df.columns: df[c] = df[c].astype("int64")
    if "rt" in df: df["rt"] = df["rt"].astype("float64")
    if "feedback" in df: df["feedback"] = df["feedback"].astype("float64")
    sdf = df[df["participant_id"] == int(subj_id)].copy()
    if sdf.empty: raise SystemExit(f"No rows for participant_id={subj_id} in {csvfile}")
    sdf = add_ushared_udiff(sdf, subj_col="participant_id")
    sdf["valid_upto"] = len(sdf)
    for c in ["response2","feedback"]:
        if c not in sdf: sdf[c] = 0 if c=="response2" else 0.0
    return sdf.reset_index(drop=True)

def build_model_single_subject(sdf):
    params = ["rl.alpha","scaler","a","z","t","theta"]
    logp = make_rldm_logp_op(n_participants=1, n_trials=len(sdf), n_params=len(params),
                             n_states=int(sdf["state2"].max())+1)
    CustomRV = make_hssm_rv(simulator_fun=create_dummy_simulator(), list_params=params)
    cfg = hssm.ModelConfig(
        response=["rt","response"], list_params=params, choices=[0,1],
        default_priors={}, bounds=dict(
            rl_alpha=(0.01,1.0), scaler=(1.0,4.0), a=(0.3,2.5),
            z=(0.1,0.9), t=(0.1,2.0), theta=(0.0,1.2)
        ),
        rv=CustomRV,
        extra_fields=["participant_id","trial_id","feedback","state1","state2",
                      "response2","valid_upto","U_shared","U_diff"],
        backend="jax",
    )
    return hssm.HSSM(
        data=sdf, model_config=cfg, p_outlier=0, lapse=None,
        loglik=logp, loglik_kind="approx_differentiable",
        noncentered=True, process_initvals=False,
        include=[
            hssm.Param("rl.alpha", "rl_alpha ~ 1",
                       {"Intercept": hssm.Prior("TruncatedNormal", lower=0.01, upper=1.0, mu=0.3)}),
            hssm.Param("scaler", "scaler ~ 1",
                       {"Intercept": hssm.Prior("TruncatedNormal", lower=1.0, upper=4.0, mu=1.5)}),
            hssm.Param("a", "a ~ 1",
                       {"Intercept": hssm.Prior("TruncatedNormal", lower=0.3, upper=2.5, mu=1.0)}),
            hssm.Param("z", "z ~ 1",
                       {"Intercept": hssm.Prior("TruncatedNormal", lower=0.1, upper=0.9, mu=0.2)}),
            hssm.Param("t", "t ~ 1 + U_shared + U_diff",
                       {"Intercept": hssm.Prior("TruncatedNormal", lower=0.01, upper=2.0, mu=0.2, initval=0.1),
                        "U_shared": hssm.Prior("Normal", mu=0.0, sigma=0.5),
                        "U_diff":   hssm.Prior("Normal", mu=0.0, sigma=0.5)}),
            hssm.Param("theta", "theta ~ 1",
                       {"Intercept": hssm.Prior("TruncatedNormal", lower=0.0, upper=1.2, mu=0.3)}),
        ],
    )

def main():
    ap = argparse.ArgumentParser(description="Single-subject regressed HSSM on CPU (no GPU).")
    ap.add_argument("--ssc", type=int, required=True, help="MTST condition (e.g., 2/3/4/5 states)")
    ap.add_argument("--subj", type=int, required=True, help="participant_id (formerly subj_idx)")
    ap.add_argument("--chain-id", type=int, default=1, help="Chain id label for saving (1-based ok)")
    ap.add_argument("--chains", type=int, default=1, help="Number of chains to sample")
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--sampler", type=str, default="nuts_numpyro")
    ap.add_argument("--outdir", type=str, default="out_reg_indiv")
    ap.add_argument("--target_accept", type=float, default=0.9)
    ap.add_argument("--max_treedepth", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    csvfile = f"hddm2_fixed_final_{args.ssc}states.csv"
    if not os.path.exists(csvfile):
        raise FileNotFoundError(f"CSV not found: {csvfile}")

    sdf = load_single_subject(csvfile, args.subj)
    model = build_model_single_subject(sdf)

    seed = (args.subj + 1000 * args.ssc) ^ (args.chain_id * 2654435761 % 2**32)

    idata = model.sample(
        sampler=args.sampler,
        chains=args.chains,
        draws=args.draws,
        tune=args.tune,
        target_accept=args.target_accept,
        random_seed=seed,
        cores=1,
        # inference_kwargs={
        #     "chain_method": "vectorized",
        #     "dense_mass": False,
        #     "max_treedepth": args.max_treedepth,
        # },
    )

    # Stamp the chain coordinate with the provided chain-id (useful when saving 1 chain per file)
    for grp in ["posterior", "sample_stats", "log_likelihood"]:
        if hasattr(idata, grp) and getattr(idata, grp) is not None:
            ds = getattr(idata, grp)
            if "chain" in ds.dims and ds.sizes["chain"] == 1:
                ds = ds.assign_coords(chain=[args.chain_id])
                setattr(idata, grp, ds)

    subdir = args.outdir
    os.makedirs(subdir, exist_ok=True)

    # Save posterior & summary with chain id
    nc_path = os.path.join(subdir, f"ssc{args.ssc}_reg_s{args.subj:04d}_chain{args.chain_id}.nc")
    az.to_netcdf(idata, nc_path)
    summ = az.summary(idata, var_names=["rl.alpha","scaler","a","z","t","theta"], hdi_prob=0.95)
    summ.to_csv(os.path.join(subdir, f"summary_reg_ssc{args.ssc}_s{args.subj:04d}_chain{args.chain_id}.csv"))

    print(f"Saved subject {args.subj} (ssc={args.ssc}) chain {args.chain_id} → {subdir}")
    print(f"InferenceData: {nc_path}")

if __name__ == "__main__":
    main()
