#!/usr/bin/env python
import argparse, os, numpy as np, pandas as pd, arviz as az
import hssm
from functools import partial
from hssm.utils import decorate_atomic_simulator
from hssm.distribution_utils.dist import make_hssm_rv
from hssm.likelihoods.rldm import make_rldm_logp_op

def make_sim():
    def sim_wrapper(simulator_fun, theta, model, n_samples, random_state, **kwargs):
        rt = np.random.uniform(0.2, 0.6, n_samples)
        ch = np.random.randint(0, 2, n_samples)
        return np.column_stack([rt, ch])
    return decorate_atomic_simulator(model_name="custom", choices=[0,1], obs_dim=2)(
        partial(sim_wrapper, simulator_fun=None, model="custom", n_samples=1)
    )

def load_and_filter(csvfile, subj):
    df = pd.read_csv(csvfile, index_col=0)
    df = df.rename(columns={"subj_idx":"participant_id","trial":"trial_id","response1":"response","rt1":"rt"})
    df["participant_id"] = df["participant_id"].astype("int64")
    sdf = df[df["participant_id"] == int(subj)].copy()
    if sdf.empty:
        raise SystemExit(f"No rows for participant_id={subj} in {csvfile}")
    # required cols / dtypes
    sdf["rt"] = sdf["rt"].astype("float64")
    for c in ["response","state1","state2","trial_id","participant_id"]:
        sdf[c] = sdf[c].astype("int64")
    # Ensure optional fields exist if your CSV lacks them
    if "feedback" not in sdf.columns:
        sdf["feedback"] = 0.0
    if "response2" not in sdf.columns:
        sdf["response2"] = 0
    # valid_upto = N (no padding needed; single subject)
    N = len(sdf)
    sdf["valid_upto"] = N
    return sdf

def build_nonhier_model(sdf: pd.DataFrame):
    n_participants = 1
    n_trials = len(sdf)
    n_states = int(sdf["state2"].max()) + 1

    logp = make_rldm_logp_op(
        n_participants=n_participants,
        n_trials=n_trials,
        n_params=7,
        n_states=n_states,
    )

    params = ["rl.alpha","scaler","a","z","t","theta","w"]
    CustomRV = make_hssm_rv(simulator_fun=make_sim(), list_params=params)

    cfg = hssm.ModelConfig(
        response=["rt","response"],
        list_params=params,
        choices=[0,1],
        default_priors={},
        bounds=dict(
            rl_alpha=(0.01,1.0),
            scaler=(1.0,4.0),
            a=(0.3,2.5),
            z=(0.1,0.9),
            t=(0.1,2.0),
            theta=(0.0,1.2),
            w=(0.1,0.9),
        ),
        rv=CustomRV,
        extra_fields=["participant_id","trial_id","feedback","state1","state2","response2","valid_upto"],
        backend="jax",
    )

    # IMPORTANT: no random-effects terms here → fully individual fit
    return hssm.HSSM(
        data=sdf,
        model_config=cfg,
        p_outlier=0,
        lapse=None,
        loglik=logp,
        loglik_kind="approx_differentiable",
        noncentered=True,
        process_initvals=False,
        include=[
            hssm.Param("rl.alpha", formula="rl_alpha ~ 1",
                       prior={"Intercept": hssm.Prior("TruncatedNormal", lower=0.01, upper=1.0,  mu=0.3)}),
            hssm.Param("scaler",   formula="scaler ~ 1",
                       prior={"Intercept": hssm.Prior("TruncatedNormal", lower=1.0,  upper=4.0,  mu=1.5)}),
            hssm.Param("a",        formula="a ~ 1",
                       prior={"Intercept": hssm.Prior("TruncatedNormal", lower=0.3,  upper=2.5,  mu=1.0)}),
            hssm.Param("z",        formula="z ~ 1",
                       prior={"Intercept": hssm.Prior("TruncatedNormal", lower=0.1,  upper=0.9,  mu=0.2)}),
            hssm.Param("t",        formula="t ~ 1",
                       prior={"Intercept": hssm.Prior("TruncatedNormal", lower=0.01, upper=2.0,  mu=0.2, initval=0.1)}),
            hssm.Param("theta",    formula="theta ~ 1",
                       prior={"Intercept": hssm.Prior("TruncatedNormal", lower=0.0,  upper=1.2,  mu=0.3)}),
            hssm.Param("w",        formula="w ~ 1",
                       prior={"Intercept": hssm.Prior("TruncatedNormal", lower=0.1,  upper=0.9,  mu=0.2)}),
        ],
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssc", type=int, required=True, help="MTST condition (e.g., 2/3/4/5 states)")
    ap.add_argument("--subj", type=int, required=True, help="participant_id (a.k.a. subj_idx)")
    ap.add_argument("--chain-id", type=int, default=1, help="Chain id label for saving (1-based ok)")
    ap.add_argument("--chains", type=int, default=1, help="Number of chains to sample")
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--sampler", type=str, default="nuts_numpyro")
    ap.add_argument("--outdir", type=str, default="out")
    ap.add_argument("--target_accept", type=float, default=0.9, help="Target acceptance rate for NUTS sampler.")
    ap.add_argument("--max_treedepth", type=int, default=12, help="Maximum tree depth for NUTS sampler.")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    csvfile = f"hddm2_fixed_final_{args.ssc}states.csv"
    sdf = load_and_filter(csvfile, args.subj)

    model = build_nonhier_model(sdf)
    # mix in chain-id so separate chain runs get different seeds
    seed = (args.subj + 1000 * args.ssc) ^ (args.chain_id * 2654435761 % 2**32)

    idata = model.sample(
        sampler=args.sampler,
        chains=args.chains,
        draws=args.draws,
        tune=args.tune,
        target_accept=args.target_accept,
        random_seed=seed,
        cores=1,                # avoid forking extra writers to stdout
        inference_kwargs={
            "chain_method": "vectorized",
            "dense_mass": False,
            "nuts_kwargs": {"max_tree_depth": args.max_treedepth},  # ← correct for NumPyro
        }
        # inference_kwargs={
        # "nuts": {"max_treedepth": args.max_treedepth},  # <-- PyMC key/name
    # },
    )

    # If this invocation ran a single chain, stamp the chain coord with chain-id
    for grp in ["posterior", "sample_stats", "log_likelihood"]:
        if hasattr(idata, grp) and getattr(idata, grp) is not None:
            ds = getattr(idata, grp)
            if "chain" in ds.dims and ds.sizes["chain"] == 1:
                ds = ds.assign_coords(chain=[args.chain_id])
                setattr(idata, grp, ds)

    # outputs per subject
    subdir = args.outdir
    os.makedirs(subdir, exist_ok=True)
    nc_path = os.path.join(subdir, f"ssc{args.ssc}_s{args.subj:04d}_chain{args.chain_id}.nc")
    az.to_netcdf(idata, nc_path)

    # quick CSV summary
    summ = az.summary(idata, var_names=["rl.alpha","scaler","a","z","t","theta","w"], hdi_prob=0.95)
    summ.to_csv(os.path.join(subdir, f"summary_ssc{args.ssc}_s{args.subj:04d}_chain{args.chain_id}.csv"))

    print(f"Saved subject {args.subj} (ssc={args.ssc}) chain {args.chain_id} → {subdir}")
    print(f"InferenceData: {nc_path}")

if __name__ == "__main__":
    main()
