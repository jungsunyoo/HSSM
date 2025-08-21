
import os
import warnings
os.environ["JAX_PLATFORM_NAME"] = "gpu"  # set before importing jax
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
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
import jax
from jax.experimental import compilation_cache
compilation_cache.compilation_cache.set_cache_dir("/tmp/jax_cache")
# os.environ["JAX_PLATFORM_NAME"] = "gpu"
# jax.config.update("jax_enable_x64", True)  # try False for speed if stable

# import warnings 
# warnings.filterwarnings( "ignore", category=FutureWarning, message=".DataFrameGroupBy\.apply operated on the grouping columns." )

def add_ushared_udiff(dataset: pd.DataFrame, subj_col: str = "participant_id",
                      alpha0: float = 1.0, beta0: float = 1.0) -> pd.DataFrame:
    """
    Compute within-subject z-scored uncertainties and rotate to U_shared/U_diff.
    Requires: state1, state2, response, participant_id.
    """
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

        K = len(pair_list)
        alpha = np.full(K, alpha0, dtype=float)
        beta  = np.full(K, beta0, dtype=float)
        var_ts = np.empty(len(g), dtype=float)
        for t, k in enumerate(pair_idx):
            a, b = alpha[k], beta[k]
            var_ts[t] = (a * b) / ((a + b)**2 * (a + b + 1.0))
            if is_common[t]:
                alpha[k] = a + 1.0
            else:
                beta[k] = b + 1.0
        return var_ts

    def _run_state(g):
        pair_idx = g["state1"].to_numpy(np.int64)
        resp = g["response"].to_numpy(np.int64)
        s2 = g["state2"].to_numpy(np.int64)
        trial_pairs = pair_list[pair_idx]
        rows = np.arange(len(g))
        chosen_state = trial_pairs[rows, resp]
        is_common = (s2 == chosen_state).astype(np.int64)

        alpha = np.full(n, alpha0, dtype=float)
        beta  = np.full(n, beta0, dtype=float)
        var_ts = np.empty(len(g), dtype=float)
        for t, k in enumerate(chosen_state):
            a, b = alpha[k], beta[k]
            var_ts[t] = (a * b) / ((a + b)**2 * (a + b + 1.0))
            if is_common[t]:
                alpha[k] = a + 1.0
            else:
                beta[k] = b + 1.0
        return var_ts

    df["uncertainty_pair"] = np.nan
    df["uncertainty_state"] = np.nan
    for pid, g in df.groupby(subj_col, sort=False):
        df.loc[g.index, "uncertainty_pair"] = _run_pair(g)
        df.loc[g.index, "uncertainty_state"] = _run_state(g)

    z = df.groupby(subj_col)[["uncertainty_pair", "uncertainty_state"]].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8)
    )
    df["U_pair_z"] = z["uncertainty_pair"]
    df["U_state_z"] = z["uncertainty_state"]
    root2 = np.sqrt(2.0)
    df["U_shared"] = (df["U_pair_z"] + df["U_state_z"]) / root2
    df["U_diff"]   = (df["U_pair_z"] - df["U_state_z"]) / root2
    return df

# Add this helper where you prepare the dataset
def add_valid_upto_and_pad(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["participant_id", "trial_id"]).reset_index(drop=True)
    counts = df.groupby("participant_id").size()
    max_len = int(counts.max())

    def pad_group(g):
        g = g.copy()
        n = int(len(g))
        g["valid_upto"] = n
        if n < max_len:
            pid = int(g["participant_id"].iloc[0])
            start_trial = int(g["trial_id"].max()) + 1
            pad_len = max_len - n
            tail = pd.DataFrame({
                "participant_id": np.full(pad_len, pid, dtype=np.int32),
                "trial_id": np.arange(start_trial, start_trial + pad_len, dtype=np.int32),
                "response": np.zeros(pad_len, dtype=np.int32),
                "response2": np.zeros(pad_len, dtype=np.int32),
                "rt": np.zeros(pad_len, dtype=np.float32),
                "feedback": np.zeros(pad_len, dtype=np.float32),
                "state1": np.zeros(pad_len, dtype=np.int32),
                "state2": np.zeros(pad_len, dtype=np.int32),
                "valid_upto": np.full(pad_len, n, dtype=np.int32),
            })
            g = pd.concat([g, tail], ignore_index=True)
        return g

    out = df.groupby("participant_id", group_keys=False).apply(pad_group).reset_index(drop=True)

    # Enforce dtypes after concat (pandas may upcast)
    for c in ["participant_id", "trial_id", "response", "response2", "state1", "state2", "valid_upto"]:
        out[c] = out[c].astype("int32")
    for c in ["rt", "feedback"]:
        out[c] = out[c].astype("float32")
    return out

def create_dummy_simulator():
    """Create a dummy simulator function for RLSSM model."""
    def sim_wrapper(simulator_fun, theta, model, n_samples, random_state, **kwargs):
        sim_rt = np.random.uniform(0.2, 0.6, n_samples)
        sim_ch = np.random.randint(0, 2, n_samples)
        return np.column_stack([sim_rt, sim_ch])

    wrapped_simulator = partial(sim_wrapper, simulator_fun=simulator, model="custom", n_samples=1)
    return decorate_atomic_simulator(model_name="custom", choices=[0, 1], obs_dim=2)(wrapped_simulator)


def build_model(dataset: pd.DataFrame):
    # Ensure required columns exist and types are correct
    if "participant_id" not in dataset.columns and "subj_idx" in dataset.columns:
        dataset = dataset.rename(columns={"subj_idx": "participant_id"})
    if "trial_id" not in dataset.columns and "trial" in dataset.columns:
        dataset = dataset.rename(columns={"trial": "trial_id"})
    if "response" not in dataset.columns and "response1" in dataset.columns:
        dataset = dataset.rename(columns={"response1": "response"})
    if "rt" not in dataset.columns and "rt1" in dataset.columns:
        dataset = dataset.rename(columns={"rt1": "rt"})

    # Cast indices/labels to integer
    for col in ("participant_id", "state1", "state2"):
        if col in dataset.columns:
            dataset[col] = dataset[col].astype("int32")

    # Infer participant counts and trials for logp op shape
    trials_per_participant = dataset.groupby("participant_id").size().tolist()
    n_participants = len(trials_per_participant)
    n_trials = max(trials_per_participant)
    
    # Determine stage-2 state count ONCE (static) and pass it to the op
    n_states = int(dataset["state2"].max()) + 1
    
    # Log-likelihood op (JAX-backed)
    logp_jax_op = make_rldm_logp_op(
        n_participants=n_participants,
        n_trials=n_trials,
        n_params=6,  # ['rl.alpha', 'scaler', 'a', 'z', 't', 'theta']
        n_states=n_states,
    )

    # RandomVariable via dummy simulator (for posterior predictive compatibility)
    list_params = ["rl.alpha", "scaler", "a", "z", "t", "theta"]
    decorated_simulator = create_dummy_simulator()
    CustomRV = make_hssm_rv(simulator_fun=decorated_simulator, list_params=list_params)

    # Model config
    model_config = hssm.ModelConfig(
        response=["rt", "response"],
        list_params=list_params,
        choices=[0, 1],
        default_priors={},
        bounds=dict(
            rl_alpha=(0.01, 1.0),
            scaler=(1.0, 4.0),
            a=(0.3, 2.5),
            z=(0.1, 0.9),
            t=(0.1, 2.0),
            theta=(0.0, 1.2),
            w=(0.1, 0.9),
        ),
        rv=CustomRV,
        extra_fields=[
            "participant_id",
            "trial_id",
            "feedback",
            "state1",
            "state2",
            "response2",
            "valid_upto",
            "U_shared",
            "U_diff",
        ],
        backend="jax",
    )

    # HSSM model
    hssm_model = hssm.HSSM(
        data=dataset,
        model_config=model_config,
        p_outlier=0,
        lapse=None,
        loglik=logp_jax_op,
        loglik_kind="approx_differentiable",
        noncentered=True,
        process_initvals=False,
        include=[
            hssm.Param(
                "rl.alpha",
                formula="rl_alpha ~ 1 + (1|participant_id)",
                prior={"Intercept": hssm.Prior("TruncatedNormal", lower=0.01, upper=1.0, mu=0.3)},
            ),
            hssm.Param(
                "scaler",
                formula="scaler ~ 1 + (1|participant_id)",
                prior={"Intercept": hssm.Prior("TruncatedNormal", lower=1.0, upper=4.0, mu=1.5)},
            ),
            hssm.Param(
                "a",
                formula="a ~ 1 + (1|participant_id)",
                prior={"Intercept": hssm.Prior("TruncatedNormal", lower=0.3, upper=2.5, mu=1.0)},
            ),
            hssm.Param(
                "z",
                formula="z ~ 1 + (1|participant_id)",
                prior={"Intercept": hssm.Prior("TruncatedNormal", lower=0.1, upper=0.9, mu=0.2)},
            ),
            hssm.Param(
                "t",
                formula="t ~ 1 + U_shared + U_diff + (1|participant_id)",  # fixed slopes + random intercept
                prior={
                    "Intercept": hssm.Prior("TruncatedNormal", lower=0.01, upper=2.0, mu=0.2, initval=0.1),
                    "U_shared": hssm.Prior("Normal", mu=0.0, sigma=0.5),
                    "U_diff":   hssm.Prior("Normal", mu=0.0, sigma=0.5),
                },
            ),
            # If you want participant‑wise slopes for U_shared and U_diff, add random slopes:
            # Compact: formula="t ~ 1 + U_shared + U_diff + (1 + U_shared + U_diff | participant_id)"

            # hssm.Param(
            #     "t",
            #     formula="t ~ 1 + U_shared + U_diff + (1|participant_id) + (U_shared|participant_id) + (U_diff|participant_id)",
            #     prior={
            #         "Intercept": hssm.Prior("TruncatedNormal", lower=0.01, upper=2.0, mu=0.2, initval=0.1),
            #         "U_shared": hssm.Prior("Normal", mu=0.0, sigma=0.5),
            #         "U_diff":   hssm.Prior("Normal", mu=0.0, sigma=0.5),
            #     },
            # ),            
            
            hssm.Param(
                "theta",
                formula="theta ~ 1 + (1|participant_id)",
                prior={"Intercept": hssm.Prior("TruncatedNormal", lower=0.0, upper=1.2, mu=0.3)},
            ),
            # hssm.Param(
            #     "w",
            #     formula="w ~ 1 + (1|participant_id)",
            #     prior={"Intercept": hssm.Prior("TruncatedNormal", lower=0.1, upper=0.9, mu=0.2)},
            # ),
        ],
    )

    return hssm_model


# ...existing code...
def main():
    parser = argparse.ArgumentParser(description="Run single MCMC chain for RLSSM two-step model.")
    # parser.add_argument("--csv", type=str, required=True)
    # parser.add_argument("--nmodel", type=int, default="rlssm_tst", help="Model name (default: rlssm_tst).")
    parser.add_argument("--ssc", type=int, required=True, help="MTST condition.")
    parser.add_argument("--chain-id", type=int, required=True, help="Chain index (0-based or 1-based).")
    # parser.add_argument("--seed", type=int, required=True, help="Random seed for this chain.")
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--sampler", type=str, default="nuts_numpyro")
    # parser.add_argument("--sampler", type=str, default="blackjax_nuts")
    parser.add_argument("--outdir", type=str, default="chains")
    parser.add_argument("--chains", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    seed = args.chain_id + 1000 * args.ssc
    # args.seed = seed  # Set the seed for reproducibility
    csvfile = 'hddm2_fixed_final_' + str(args.ssc) + 'states.csv'

    # if not os.path.exists(args.csv):
    if not os.path.exists(csvfile):    
        raise FileNotFoundError(f"CSV not found: ssc {args.ssc}")

    dataset = pd.read_csv(csvfile, index_col=0)



    dataset.rename(columns={'subj_idx': 'participant_id'}, inplace=True)
    dataset.rename(columns={'trial': 'trial_id'}, inplace=True)
    dataset.rename(columns={'response1': 'response'}, inplace=True)
    dataset.rename(columns={'rt1': 'rt'}, inplace=True)

    dataset["rt"] = dataset["rt"].astype('float32')
    dataset["response"] = dataset["response"].astype('int32')   
    dataset["feedback"] = dataset["feedback"].astype('int32')
    # dataset["valid_upto"] = dataset["valid_upto"].astype('int32')
    dataset["trial_id"] = dataset["trial_id"].astype('int32')
    dataset["state1"] = dataset["state1"].astype('int32')
    dataset["state2"] = dataset["state2"].astype('int32')
    dataset["response2"] = dataset["response2"].astype('int32')
    dataset["participant_id"] = dataset["participant_id"].astype('int32')

    # Compute U_shared/U_diff
    dataset = add_ushared_udiff(dataset, subj_col="participant_id")

    dataset = add_valid_upto_and_pad(dataset)

    # Fill padded rows (outside valid_upto) with 0 for regressors and enforce float32 dtype
    for c in ["U_shared", "U_diff"]:
        if c in dataset.columns:
            dataset[c] = dataset[c].fillna(0.0).astype("float32")
    # if "participant_id" in dataset.columns:
    #     full_ids = (
    #         dataset.groupby("participant_id")
    #         .size()
    #         .pipe(lambda s: s[s == dataset.groupby("participant_id").size().max()].index)
    #     )
    #     dataset = dataset.loc[dataset["participant_id"].isin(full_ids)].reset_index(drop=True)

    model = build_model(dataset)

    idata = model.sample(
        sampler=args.sampler,
        chains=args.chains,                    # try 2–4
        # chain_method="vectorized",   # best on single GPU
        draws=args.draws,
        tune=args.tune,              # don’t overshoot
        target_accept=0.9,          # faster if still stable
        random_seed=seed,
        cores=1,                # avoid forking extra writers to stdout
        inference_kwargs={
            "chain_method": "vectorized",
            "dense_mass": False,              # stay diagonal (much cheaper per step)
            "max_treedepth": 12,              # optional: cap runaway trees
            # you can also pass nuts_kwargs here in some versions:
            # "nuts_kwargs": {"dense_mass": False, "max_tree_depth": 12}
        },
    )
    # Re-label chain coordinate explicitly (optional consistency)
    for group in ["posterior", "sample_stats", "log_likelihood"]:
        if hasattr(idata, group) and getattr(idata, group) is not None:
            ds = getattr(idata, group)
            if "chain" in ds.dims:
                ds = ds.assign_coords(chain=[args.chain_id])
                setattr(idata, group, ds)

    outfile = os.path.join(
        args.outdir,
        f"model1_mb_reg_ssc{args.ssc}_idata_chain{args.chain_id}.nc"
    )
    az.to_netcdf(idata, outfile)
    print(f"Saved ssc {args.ssc} chain {args.chain_id} to {outfile}")

if __name__ == "__main__":
    main()