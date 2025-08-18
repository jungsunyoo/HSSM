
import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"  # set before importing jax
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import argparse
import numpy as np
import pandas as pd
import arviz as az

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
jax.config.update("jax_enable_x64", True)  # try False for speed if stable



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
                "participant_id": pid,
                "trial_id": np.arange(start_trial, start_trial + pad_len, dtype=int),
                "response": 0,
                "response2": 0,
                "rt": 0.0,
                "feedback": 0.0,
                "state1": 0,
                "state2": 0,
                "valid_upto": n,
            })
            g = pd.concat([g, tail], ignore_index=True)
        return g

    out = df.groupby("participant_id", group_keys=False).apply(pad_group).reset_index(drop=True)
    for c in ["participant_id", "trial_id", "response", "response2", "state1", "state2", "valid_upto"]:
        if c in out.columns:
            out[c] = out[c].astype("int64")
        else:
            out[c] = out[c].astype("float64")  # ensure all are float64
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
            dataset[col] = dataset[col].astype("int64")

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
        n_params=7,  # ['rl.alpha', 'scaler', 'a', 'z', 't', 'theta', 'w']
        n_states=n_states,
    )

    # RandomVariable via dummy simulator (for posterior predictive compatibility)
    list_params = ["rl.alpha", "scaler", "a", "z", "t", "theta", "w"]
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
                formula="t ~ 1 + (1|participant_id)",
                prior={
                    "Intercept": hssm.Prior(
                        "TruncatedNormal", lower=0.01, upper=2.0, mu=0.2, initval=0.1
                    )
                },
            ),
            hssm.Param(
                "theta",
                formula="theta ~ 1 + (1|participant_id)",
                prior={"Intercept": hssm.Prior("TruncatedNormal", lower=0.0, upper=1.2, mu=0.3)},
            ),
            hssm.Param(
                "w",
                formula="w ~ 1 + (1|participant_id)",
                prior={"Intercept": hssm.Prior("TruncatedNormal", lower=0.1, upper=0.9, mu=0.2)},
            ),
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

    dataset["rt"] = dataset["rt"].astype('float64')
    dataset["response"] = dataset["response"].astype('int64')

    dataset['state1']=dataset['state1'].astype('int64')
    dataset['state2']=dataset['state2'].astype('int64')
    dataset['participant_id']=dataset['participant_id'].astype('int64')
    
    dataset = add_valid_upto_and_pad(dataset)
    
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
        chains=1,                    # try 2–4
        # chain_method="vectorized",   # best on single GPU
        draws=args.draws,
        tune=args.tune,              # don’t overshoot
        target_accept=0.9,          # faster if still stable
        random_seed=seed,
        
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
        f"ssc{args.ssc}_idata_chain{args.chain_id}.nc"
    )
    az.to_netcdf(idata, outfile)
    print(f"Saved ssc {args.ssc} chain {args.chain_id} to {outfile}")

if __name__ == "__main__":
    main()