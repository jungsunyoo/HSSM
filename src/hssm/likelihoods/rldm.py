"""The log-likelihood function for the RLDM model."""

from typing import Callable

import jax
import jax.numpy as jnp
from jax.lax import dynamic_slice, scan
from jax.scipy.special import logsumexp
from jax.lax import dynamic_slice_in_dim, scan
from ..distribution_utils.func_utils import make_vjp_func
from ..distribution_utils.jax import make_jax_logp_ops

# from ..onnx_utils.model import download_hf
from ..distribution_utils.onnx import (
    make_jax_logp_funcs_from_onnx,
    make_jax_matrix_logp_funcs_from_onnx,
)

rlssm_model_config_list = {
    "rlssm1": {
        "name": "rlssm1",
        "description": "RLSSM model where the learning process is a simple  \
                        Rescorla-Wagner model in a 2-armed bandit task. \
                        The decision process is a collapsing bound DDM (angle model).",
        "n_params": 6,
        "n_extra_fields": 3,
        "list_params": ["rl.alpha", "scaler", "a", "Z", "t", "theta"],
        "extra_fields": ["participant_id", "trial_id", "feedback"],
        "decision_model": "LAN",
        "LAN": "angle",
    },
    "rlssm2": {
        "name": "rlssm2",
        "description": "RLSSM model where the learning process is a simple \
                        Rescorla-Wagner model in a 2-armed bandit task. \
                        The decision process is a collapsing bound DDM (angle model). \
                        Same as rlssm1, but with dual learning rates for positive and \
                         negative prediction errors. \
                        This model is meant to serve as a tutorial for showing how to  \
                         implement a custom RLSSM model in HSSM.",
        "n_params": 7,
        "n_extra_fields": 3,
        "list_params": ["rl.alpha", "rl.alpha_neg", "scaler", "a", "Z", "t", "theta"],
        "extra_fields": ["participant_id", "trial_id", "feedback"],
        "decision_model": "LAN",
        "LAN": "angle",
    },
    "rlssm-tst": {
        "name": "rlssm-tst",
        "description": "TWO STAGE TASK. RLSSM model where the learning process is a simple \
                        Rescorla-Wagner model in a 2-armed bandit task. \
                        The decision process is a collapsing bound DDM (angle model). \
                        Same as rlssm1, but with dual learning rates for positive and \
                         negative prediction errors. \
                        This model is meant to serve as a tutorial for showing how to  \
                         implement a custom RLSSM model in HSSM.",
        "n_params": 7,
        "n_extra_fields": 8,
        "list_params": ['rl.alpha', 'scaler', 'a', 'Z', 't', 'theta', 'w'],
        "extra_fields": ["participant_id", "trial_id", "feedback", "state1", "state2", "response2", "valid_upto", "n_states"],
        "decision_model": "LAN",
        "LAN": "angle",
    },    
    "rlwmssm_v2": {
        "name": "rlwmssm_v2",
        "description": "RLSSM model where the learning process is the RLWM model \
                        (see Collins & Frank, 2012 for details).  \
                        The decision process is a collapsing bound LBA \
                            (LBA angle model). ",
        "n_params": 10,
        "n_extra_fields": 6,
        "list_params": [
            "a",
            "z",
            "theta",
            "alpha",
            "phi",
            "rho",
            "gamma",
            "epsilon",
            "C",
            "eta",
        ],
        "extra_fields": [
            "participant_id",
            "set_size",
            "stimulus_id",
            "feedback",
            "new_block_start",
            "unidim_mask",
        ],
        "decision_model": "LAN",
        "LAN": "dev_lba_angle_3_v2",
    },
}

MODEL_NAME = "rlssm-tst"
MODEL_CONFIG = rlssm_model_config_list[MODEL_NAME]

if not isinstance(MODEL_CONFIG["n_extra_fields"], int) or not isinstance(
    MODEL_CONFIG["n_params"], int
):
    raise ValueError(
        f"Expected 'n_extra_fields' to be an int, \
            got {type(MODEL_CONFIG['n_extra_fields'])}."
    )
num_params = int(MODEL_CONFIG["n_params"])
num_extra_fields = int(MODEL_CONFIG["n_extra_fields"])
total_params = num_params + num_extra_fields


lan_logp_jax_func = make_jax_matrix_logp_funcs_from_onnx(
    model="angle.onnx",
)

jax_LAN_logp = make_jax_logp_funcs_from_onnx(
    "../../tests/fixtures/dev_lba_angle_3_v2.onnx",
    [True] * 6,
)[0]


# RLSSM model likelihood function
def rlssm1_logp_inner_func(
    subj,
    ntrials_subj,
    data,
    rl_alpha,
    scaler,
    a,
    z,
    t,
    theta,
    trial,
    feedback,
):
    """Compute the log likelihood for a given subject using the RLDM model."""
    rt = data[:, 0]
    response = data[:, 1]

    subj = jnp.astype(subj, jnp.int32)

    # Extracting the parameters and data for the specific subject
    subj_rl_alpha = dynamic_slice(rl_alpha, [subj * ntrials_subj], [ntrials_subj])
    subj_scaler = dynamic_slice(scaler, [subj * ntrials_subj], [ntrials_subj])
    subj_a = dynamic_slice(a, [subj * ntrials_subj], [ntrials_subj])
    subj_z = dynamic_slice(z, [subj * ntrials_subj], [ntrials_subj])
    subj_t = dynamic_slice(t, [subj * ntrials_subj], [ntrials_subj])
    subj_theta = dynamic_slice(theta, [subj * ntrials_subj], [ntrials_subj])

    subj_trial = dynamic_slice(trial, [subj * ntrials_subj], [ntrials_subj])
    subj_response = dynamic_slice(response, [subj * ntrials_subj], [ntrials_subj])
    subj_rt = dynamic_slice(rt, [subj * ntrials_subj], [ntrials_subj])
    subj_feedback = dynamic_slice(feedback, [subj * ntrials_subj], [ntrials_subj])

    # Initialize the LAN matrix that will hold the trial-by-trial data
    # The matrix will have 7 columns: data (choice, rt) and parameters of
    # the angle model (v, a, z, t, theta)
    # The number of rows is equal to the number of trials for the subject
    q_val = jnp.ones(2) * 0.5
    LAN_matrix_init = jnp.zeros((ntrials_subj, 7))

    # function to process each trial
    def process_trial(carry, inputs):
        q_val, loglik, LAN_matrix, t = carry
        state, action, rt, reward = inputs
        state = jnp.astype(state, jnp.int32)
        action = jnp.astype(action, jnp.int32)

        # drift rate on each trial depends on difference in expected rewards for
        # the two alternatives:
        # drift rate = (q_up - q_low) * scaler where
        # the scaler parameter describes the weight to put on the difference in
        # q-values.
        computed_v = (q_val[1] - q_val[0]) * subj_scaler[t]

        # compute the reward prediction error
        delta_RL = reward - q_val[action]

        # update the q-values using the RL learning rule (here, simple TD rule)
        q_val = q_val.at[action].set(q_val[action] + subj_rl_alpha[t] * delta_RL)

        # update the LAN_matrix with the current trial data
        # The first column is the drift rate, followed by
        # the parameters a, z, t, theta, rt, and action
        segment_result = jnp.array(
            [computed_v, subj_a[t], subj_z[t], subj_t[t], subj_theta[t], rt, action]
        )
        LAN_matrix = LAN_matrix.at[t, :].set(segment_result)

        return (q_val, loglik, LAN_matrix, t + 1), None

    trials = (
        subj_trial,
        subj_response,
        subj_rt,
        subj_feedback,
    )
    (q_val, LL, LAN_matrix, _), _ = scan(
        process_trial, (q_val, 0.0, LAN_matrix_init, 0), trials
    )

    # forward pass through the LAN to compute log likelihoods
    LL = lan_logp_jax_func(LAN_matrix)

    return LL.ravel()


# RLSSM model liklihood function
def rlssm2_logp_inner_func(
    subj,
    ntrials_subj,
    data,
    rl_alpha,
    rl_alpha_neg,
    scaler,
    a,
    z,
    t,
    theta,
    trial,
    feedback,
):
    """Compute the log likelihood for a given subject using the RLDM model."""
    rt = data[:, 0]
    response = data[:, 1]

    subj = jnp.astype(subj, jnp.int32)

    # Extracting the parameters and data for the specific subject
    subj_rl_alpha = dynamic_slice(rl_alpha, [subj * ntrials_subj], [ntrials_subj])
    subj_rl_alpha_neg = dynamic_slice(
        rl_alpha_neg, [subj * ntrials_subj], [ntrials_subj]
    )
    subj_scaler = dynamic_slice(scaler, [subj * ntrials_subj], [ntrials_subj])
    subj_a = dynamic_slice(a, [subj * ntrials_subj], [ntrials_subj])
    subj_z = dynamic_slice(z, [subj * ntrials_subj], [ntrials_subj])
    subj_t = dynamic_slice(t, [subj * ntrials_subj], [ntrials_subj])
    subj_theta = dynamic_slice(theta, [subj * ntrials_subj], [ntrials_subj])

    subj_trial = dynamic_slice(trial, [subj * ntrials_subj], [ntrials_subj])
    subj_response = dynamic_slice(response, [subj * ntrials_subj], [ntrials_subj])
    subj_rt = dynamic_slice(rt, [subj * ntrials_subj], [ntrials_subj])
    subj_feedback = dynamic_slice(feedback, [subj * ntrials_subj], [ntrials_subj])

    # Initialize the LAN matrix that will hold the trial-by-trial data
    # The matrix will have 7 columns: data (choice, rt) and parameters of
    # the angle model (v, a, z, t, theta)
    # The number of rows is equal to the number of trials for the subject
    q_val = jnp.ones(2) * 0.5
    LAN_matrix_init = jnp.zeros((ntrials_subj, 7))

    # function to process each trial
    def process_trial(carry, inputs):
        q_val, loglik, LAN_matrix, t = carry
        state, action, rt, reward = inputs
        state = jnp.astype(state, jnp.int32)
        action = jnp.astype(action, jnp.int32)

        # drift rate on each trial depends on difference in expected rewards for
        # the two alternatives:
        # drift rate = (q_up - q_low) * scaler where
        # the scaler parameter describes the weight to put on the difference in
        # q-values.
        computed_v = (q_val[1] - q_val[0]) * subj_scaler[t]

        # compute the reward prediction error
        delta_RL = reward - q_val[action]

        # if delta_RL < 0, use learning rate subj_rl_alpha_neg[t]
        # else use subj_rl_alpha[t]
        rl_alpha_t = jnp.where(delta_RL < 0, subj_rl_alpha_neg[t], subj_rl_alpha[t])

        # update the q-values using the RL learning rule (here, simple TD rule)
        q_val = q_val.at[action].set(q_val[action] + rl_alpha_t * delta_RL)

        # update the LAN_matrix with the current trial data
        # The first column is the drift rate, followed by
        # the parameters a, z, t, theta, rt, and action
        segment_result = jnp.array(
            [computed_v, subj_a[t], subj_z[t], subj_t[t], subj_theta[t], rt, action]
        )
        LAN_matrix = LAN_matrix.at[t, :].set(segment_result)

        return (q_val, loglik, LAN_matrix, t + 1), None

    trials = (
        subj_trial,
        subj_response,
        subj_rt,
        subj_feedback,
    )
    (q_val, LL, LAN_matrix, _), _ = scan(
        process_trial, (q_val, 0.0, LAN_matrix_init, 0), trials
    )

    # forward pass through the LAN to compute log likelihoods
    LL = lan_logp_jax_func(LAN_matrix)

    return LL.ravel()

#         "state1", 
#         "state2", 
#         "response2"

# RLSSM model liklihood function
def rlssm2_tst_1step_logp_inner_func(
    subj,
    ntrials_subj,
    data,
    rl_alpha,
    # rl_alpha_neg,
    scaler,
    a,
    z,
    t,
    theta,
    w,
    trial,
    feedback,
    state1, 
    state2, 
    response2, 
    valid_upto, 
    n_states
):
    """Compute the log likelihood for a given subject using the RLDM model."""
    rt = data[:, 0]
    response  = data[:, 1]
    
    # response2 = data[:, 2]  # New response2 column
    # state1 = data[:, 3]  # New state1 column
    # state2 = data[:, 4]  # New state2 column
    
    
    

    # subj = jnp.astype(subj, jnp.int32)
    subj = jnp.asarray(subj, dtype=jnp.int32)

    # Extracting the parameters and data for the specific subject
    subj_rl_alpha = dynamic_slice(rl_alpha, [subj * ntrials_subj], [ntrials_subj])
    # subj_rl_alpha_neg = dynamic_slice(
    #     rl_alpha_neg, [subj * ntrials_subj], [ntrials_subj]
    # )
    subj_scaler = dynamic_slice(scaler, [subj * ntrials_subj], [ntrials_subj])
    subj_a = dynamic_slice(a, [subj * ntrials_subj], [ntrials_subj])
    subj_z = dynamic_slice(z, [subj * ntrials_subj], [ntrials_subj])
    subj_t = dynamic_slice(t, [subj * ntrials_subj], [ntrials_subj])
    subj_theta = dynamic_slice(theta, [subj * ntrials_subj], [ntrials_subj])
    subj_w = dynamic_slice(w, [subj * ntrials_subj], [ntrials_subj])
    subj_trial = dynamic_slice(trial, [subj * ntrials_subj], [ntrials_subj])
    subj_response = dynamic_slice(response, [subj * ntrials_subj], [ntrials_subj])
    subj_rt = dynamic_slice(rt, [subj * ntrials_subj], [ntrials_subj])
    subj_feedback = dynamic_slice(feedback, [subj * ntrials_subj], [ntrials_subj])
    
    subj_response2 = dynamic_slice(response2, [subj * ntrials_subj], [ntrials_subj])
    subj_state1 = dynamic_slice(state1, [subj * ntrials_subj], [ntrials_subj])
    subj_state2 = dynamic_slice(state2, [subj * ntrials_subj], [ntrials_subj])    
 
     # valid_upto is constant per subject; take the first element of this block
    subj_valid_upto = dynamic_slice(valid_upto, [subj * ntrials_subj], [1])[0].astype(jnp.int32)
   

    # Initialize the LAN matrix that will hold the trial-by-trial data
    # The matrix will have 7 columns: data (choice, rt) and parameters of
    # the angle model (v, a, z, t, theta)
    # The number of rows is equal to the number of trials for the subject
    # q_val = jnp.ones(2) * 0.5
    # Q-values: first stage (2 actions), second stage (2 states x 2 actions)
    # n = jnp.unique(state2)
    # n = (jnp.max(subj_state2.astype(jnp.int32)) + 1).astype(jnp.int32)

    
    # nstates = n * (n - 1) // 2  # n = 2, so nstates = 1
    # q_val_stage1 = jnp.ones(2) * 0.5
    q_val_stage1 = jnp.ones(((n_states * (n_states-1)//2),2), dtype=jnp.float32) * 0.5
    q_val_stage2 = jnp.ones((n_states, 2), dtype=jnp.float32) * 0.5  # shape: [n_states, n_actions]
    
    LAN_matrix_init = jnp.zeros((ntrials_subj, 7))
    # LAN_matrix_init = jnp.zeros((ntrials_subj, 10))  # v1, v2, a, z, t, theta, rt, action1, action2, state2
   
    # Transition matrix: p(s2|a1)
    # Standard Daw task: action 0 -> state 0 with 0.7, state 1 with 0.3; action 1 reversed
    trans_mat = jnp.array([[0.7, 0.3], [0.3, 0.7]])  # shape (2, 2)
    
    # Per-trial update with skip for t >= valid_upto
    def _active_step(payload):
        q1, q2, loglik, LAN, t, s1_t, a1_t, a2_t, s2_t, rt_t, r_t = payload
        s1_t = jnp.asarray(s1_t, dtype=jnp.int32)
        a1_t = jnp.asarray(a1_t, dtype=jnp.int32)
        a2_t = jnp.asarray(a2_t, dtype=jnp.int32)
        s2_t = jnp.asarray(s2_t, dtype=jnp.int32)

        q_mb = jnp.array([
            trans_mat[0, 0] * q2[0, :].max() + trans_mat[0, 1] * q2[1, :].max(),
            trans_mat[1, 0] * q2[0, :].max() + trans_mat[1, 1] * q2[1, :].max(),
        ])
        q1_ = q1[s1_t, :].max(axis=0)  # shape (2,)
        net_q = subj_w[t] * q_mb + (1.0 - subj_w[t]) * q1_
        v1 = (net_q[1] - net_q[0]) * subj_scaler[t]

        delta2 = r_t - q2[s2_t, a2_t]
        q2 = q2.at[s2_t, a2_t].add(subj_rl_alpha[t] * delta2)

        delta1 = q2[s2_t, a2_t] - q1[s1_t, a1_t]
        q1 = q1.at[s1_t, a1_t].add(subj_rl_alpha[t] * delta1)

        row = jnp.array([v1, subj_a[t], subj_z[t], subj_t[t], subj_theta[t], rt_t, a1_t])
        LAN = LAN.at[t, :].set(row)

        return (q1, q2, loglik, LAN, t + 1)

    def _inactive_step(payload):
        q1, q2, loglik, LAN, t, *_ = payload
        return (q1, q2, loglik, LAN, t + 1)

    def process_trial(carry, inputs):
        q1, q2, loglik, LAN, t = carry
        s1_t, a1_t, a2_t, s2_t, rt_t, r_t = inputs
        active = t < subj_valid_upto
        payload = (q1, q2, loglik, LAN, t, s1_t, a1_t, a2_t, s2_t, rt_t, r_t)
        q1, q2, loglik, LAN, t_next = jax.lax.cond(active, _active_step, _inactive_step, payload)
        return (q1, q2, loglik, LAN, t_next), None
        
    trials = (
        subj_state1,      # s1
        subj_response,    # a1
        subj_response2,   # a2
        subj_state2,      # s2
        subj_rt,
        subj_feedback,
    )

    
    (q_val_stage1, q_val_stage2, LL, LAN_matrix, _), _ = scan(
        process_trial, (q_val_stage1, q_val_stage2, 0.0, LAN_matrix_init, 0), trials
    )    


    # forward pass through the LAN to compute log likelihoods
    LL = lan_logp_jax_func(LAN_matrix)
    # Zero-out padded trials (indices >= valid_upto)
    valid_mask = (jnp.arange(ntrials_subj) < subj_valid_upto).astype(LL.dtype)    
    LL = LL * valid_mask
    return LL.ravel()
    # return jnp.sum(LL)
    
    
# import jax
# import jax.numpy as jnp
# from jax.lax import dynamic_slice_in_dim, scan

# ------------------------------------------------------------------------
# Inner likelihood  (returns (n_trials,)  log-likelihood vector)
# ------------------------------------------------------------------------
# def rlssm2_tst_1step_logp_inner_func(
#     subj,
#     ntrials_subj,
#     data,
#     rl_alpha,
#     scaler,
#     a,
#     z,
#     t0,
#     theta,
#     w,
#     trial,     # not used inside but kept for signature parity
#     feedback,
#     state1,
#     state2,
#     response2,
# ):
#     """Vector of log-likelihoods for one subject (length = ntrials_subj)."""

#     # ---------- helper: slice this subject’s window ----------------------
#     def slc(arr):
#         start = jax.lax.convert_element_type(subj * ntrials_subj, jnp.int32)
#         return dynamic_slice_in_dim(arr, start, ntrials_subj, axis=0)

#     # ------------- per-trial observed variables --------------------------
#     rt  = data[:, 0]
#     a1  = data[:, 1]              # first-stage choice

#     # ------------- per-trial parameters & covariates ---------------------
#     rl_a   = slc(rl_alpha)
#     sclr   = slc(scaler)
#     a_s    = slc(a)
#     z_s    = slc(z)
#     t0_s   = slc(t0)
#     th_s   = slc(theta)
#     w_s    = slc(w)

#     a2 = slc(response2)           # second-stage choice
#     s1 = slc(state1)
#     s2 = slc(state2)
#     rew = slc(feedback)

#     # ---------------------- model state ----------------------------------
#     q1 = jnp.full(2,     0.5)     # first stage Q
#     q2 = jnp.full((2, 2), 0.5)    # second stage Q

#     P = jnp.array([[0.7, 0.3],    # transition matrix
#                    [0.3, 0.7]])

#     LAN0 = jnp.zeros((ntrials_subj, 7))  # v, a, z, t0, θ, rt, a1

#     # ---------------- per-trial update fn (for scan) ---------------------
#     def step(carry, inp):
#         q1_, q2_, LAN, t = carry
#         s1_t, a1_t, a2_t, s2_t, rt_t, r_t = inp

#         s1_t = s1_t.astype(jnp.int32)
#         a1_t = a1_t.astype(jnp.int32)
#         a2_t = a2_t.astype(jnp.int32)
#         s2_t = s2_t.astype(jnp.int32)

#         # model-based Q
#         q_mb = jnp.stack([
#             P[0, 0] * q2_[0].max() + P[0, 1] * q2_[1].max(),
#             P[1, 0] * q2_[0].max() + P[1, 1] * q2_[1].max(),
#         ])

#         net_q = w_s[t] * q_mb + (1.0 - w_s[t]) * q1_
#         drift = (net_q[1] - net_q[0]) * sclr[t]

#         # TD updates
#         delta2 = r_t - q2_[s2_t, a2_t]
#         q2_ = q2_.at[s2_t, a2_t].add(rl_a[t] * delta2)

#         delta1 = q2_[s2_t, a2_t] - q1_[a1_t]
#         q1_ = q1_.at[a1_t].add(rl_a[t] * delta1)

#         # build LAN row
#         LAN_row = jnp.stack(
#             [drift, a_s[t], z_s[t], t0_s[t], th_s[t], rt_t, a1_t]
#         )
#         LAN = LAN.at[t, :].set(LAN_row)

#         return (q1_, q2_, LAN, t + 1), None

#     inputs = (s1, a1, a2, s2, rt, rew)
#     (_, _, LAN_mat, _), _ = scan(step, (q1, q2, LAN0, 0), inputs)

#     # forward through LAN → (n_trials,) log-lik vector
#     return lan_logp_jax_func(LAN_mat)


# auxiliary function for the RLWMSSM model
def jax_call_LAN(LAN_matrix, unidim_mask):
    """
    Call the LAN log likelihood function with the LAN matrix and unidim_mask.

    The unidim_mask is used to mask out the log likelihoods for the
    flagged unidimensional trials.
    """
    net_input = jnp.array(LAN_matrix)
    LL = jax_LAN_logp(
        net_input[:, 6:8],
        net_input[:, 0],
        net_input[:, 1],
        net_input[:, 2],
        net_input[:, 3],
        net_input[:, 4],
        net_input[:, 5],
    )

    LL = jnp.multiply(LL, (1 - unidim_mask))

    return LL


# auxiliary function for the RLWMSSM model
def jax_softmax(q_values, beta):
    """Compute the softmax of q_values with temperature beta."""
    return jnp.exp(beta * q_values - logsumexp(beta * q_values))


# RLSSM model likelihood function
def rlwmssm_v2_inner_func(
    subj,
    ntrials_subj,
    data,
    a,
    z,
    theta,
    alpha,
    phi,
    rho,
    gamma,
    epsilon,
    C,
    eta,
    participant_id,
    set_size,
    stimulus_id,
    feedback,
    new_block_start,
    unidim_mask,
):
    """Compute the log likelihood for a given subject using the RLWMSSM model."""
    rt = data[:, 0]
    response = data[:, 1]

    num_actions = 3
    beta = 100
    subj = jnp.astype(subj, jnp.int32)

    def init_block(bl_set_size, subj_rho, subj_C):
        max_set_size = 5
        set_size_mask = jnp.arange(max_set_size) >= bl_set_size
        set_size_mask = set_size_mask[:, None]

        q_RL = jnp.ones((max_set_size, num_actions)) / num_actions
        q_RL = jnp.where(set_size_mask, -1000.0, q_RL)

        q_WM = jnp.ones((max_set_size, num_actions)) / num_actions
        q_WM = jnp.where(set_size_mask, -1000.0, q_WM)

        weight = subj_rho * jnp.minimum(1, subj_C / bl_set_size)

        return q_RL, q_WM, weight

    def update_q_values(carry, inputs):
        q_RL, q_WM, alpha, gamma, phi, num_actions = carry
        state, action, reward = inputs

        delta_RL = reward - q_RL[state, action]
        delta_WM = reward - q_WM[state, action]

        RL_alpha_factor = jnp.where(reward == 1, alpha, gamma * alpha)
        WM_alpha_factor = jnp.where(reward == 1, 1.0, gamma)

        q_RL = q_RL.at[state, action].set(
            q_RL[state, action] + RL_alpha_factor * delta_RL
        )
        q_WM = q_WM.at[state, action].set(
            q_WM[state, action] + WM_alpha_factor * delta_WM
        )

        q_WM = q_WM + phi * ((1 / num_actions) - q_WM)

        return q_RL, q_WM

    # Extracting the parameters for the specific subject
    subj_a = dynamic_slice(a, [subj * ntrials_subj], [ntrials_subj])
    subj_z = dynamic_slice(z, [subj * ntrials_subj], [ntrials_subj])
    subj_theta = dynamic_slice(theta, [subj * ntrials_subj], [ntrials_subj])
    subj_alpha = dynamic_slice(alpha, [subj * ntrials_subj], [ntrials_subj])
    subj_alpha = jnp.exp(subj_alpha)
    subj_phi = dynamic_slice(phi, [subj * ntrials_subj], [ntrials_subj])
    subj_rho = dynamic_slice(rho, [subj * ntrials_subj], [ntrials_subj])
    subj_gamma = dynamic_slice(gamma, [subj * ntrials_subj], [ntrials_subj])
    subj_epsilon = dynamic_slice(epsilon, [subj * ntrials_subj], [ntrials_subj])
    subj_C = dynamic_slice(C, [subj * ntrials_subj], [ntrials_subj])
    subj_eta = dynamic_slice(eta, [subj * ntrials_subj], [ntrials_subj])

    # Extracting the data for the specific subject
    subj_set_size = dynamic_slice(set_size, [subj * ntrials_subj], [ntrials_subj])
    subj_stimulus_id = dynamic_slice(stimulus_id, [subj * ntrials_subj], [ntrials_subj])
    subj_response = dynamic_slice(response, [subj * ntrials_subj], [ntrials_subj])
    subj_feedback = dynamic_slice(feedback, [subj * ntrials_subj], [ntrials_subj])
    subj_new_block_start = dynamic_slice(
        new_block_start, [subj * ntrials_subj], [ntrials_subj]
    )
    subj_unidim_mask = dynamic_slice(unidim_mask, [subj * ntrials_subj], [ntrials_subj])
    subj_rt = dynamic_slice(rt, [subj * ntrials_subj], [ntrials_subj])

    LAN_matrix_init = jnp.zeros((ntrials_subj, 8))

    def process_trial(carry, inputs):
        q_RL, q_WM, weight, LL, LAN_matrix, t = carry
        bl_set_size, state, action, rt, reward, new_block = inputs
        state = jnp.astype(state, jnp.int32)
        action = jnp.astype(action, jnp.int32)

        q_RL, q_WM, weight = jax.lax.cond(
            new_block == 1,
            lambda _: init_block(bl_set_size, subj_rho[t], subj_C[t]),
            lambda _: (q_RL, q_WM, weight),
            None,
        )

        pol_RL = jax_softmax(q_RL[state, :], beta)
        pol_WM = jax_softmax(q_WM[state, :], beta)

        pol = weight * pol_WM + (1 - weight) * pol_RL
        pol = (
            subj_epsilon[t] * (jnp.ones_like(pol) * 1 / num_actions)
            + (1 - subj_epsilon[t]) * pol
        )
        pol_final = pol * subj_eta[t]

        q_RL, q_WM = update_q_values(
            (q_RL, q_WM, subj_alpha[t], subj_gamma[t], subj_phi[t], num_actions),
            (state, action, reward),
        )

        LAN_matrix = LAN_matrix.at[t, :].set(
            jnp.array(
                [
                    pol_final[0],
                    pol_final[1],
                    pol_final[2],
                    subj_a[t],
                    subj_z[t],
                    subj_theta[t],
                    rt,
                    action,
                ]
            )
        )

        return (q_RL, q_WM, weight, LL, LAN_matrix, t + 1), None

    q_RL, q_WM, weight = init_block(5, 0, 5)
    trials = (
        subj_set_size,
        subj_stimulus_id,
        subj_response,
        subj_rt,
        subj_feedback,
        subj_new_block_start,
    )
    (q_RL, q_WM, weight, LL, LAN_matrix, _), _ = jax.lax.scan(
        process_trial, (q_RL, q_WM, weight, 0.0, LAN_matrix_init, 0), trials
    )

    LL = jax_call_LAN(LAN_matrix, subj_unidim_mask)

    return LL.ravel()


rldm_logp_inner_func_vmapped = jax.vmap(
    # rlssm1_logp_inner_func,
    rlssm2_tst_1step_logp_inner_func,
    in_axes=[0] + [None] * (total_params + 1),
)


def vec_logp(*args):
    """Parallelize (vectorize) the likelihood computation across subjects.

    'subj_index' arg to the JAX likelihood should be vectorized.
    """
    output = rldm_logp_inner_func_vmapped(*args).ravel()

    return output

# def make_logp_func(n_participants: int, n_trials: int) -> Callable:
#     """Create a log likelihood function for the RLDM model.

#     Parameters
#     ----------
#     n_participants : int
#         The number of participants in the dataset.
#     n_trials : int
#         The number of trials per participant.

#     Returns
#     -------
#     callable
#         A function that computes the log likelihood for the RLDM model.
#     """

#     # Ensure parameters are correctly extracted and passed to your custom function.
#     def logp(data, *dist_params) -> jnp.ndarray:
#         """Compute the log likelihood for the RLDM model.

#         Parameters
#         ----------
#         dist_params
#             A tuple containing the subject index, number of trials per subject,
#             data, and model parameters. In this case, it is expected to be
#             (rl_alpha, scaler, a, z, t, theta, trial, feedback).

#         Returns
#         -------
#         jnp.ndarray
#             The log likelihoods for each subject.
#         """
#         # Extract extra fields (adjust indices based on your model)
#         participant_id = dist_params[num_params]
#         trial = dist_params[num_params + 1]
#         feedback = dist_params[num_params + 2]

#         subj = jnp.unique(participant_id, size=n_participants).astype(jnp.int32)

#         # create parameter arrays to be passed to the likelihood function
#         rl_alpha, scaler, a, z, t, theta = dist_params[:num_params]

#         # pass the parameters and data to the likelihood function
#         return vec_logp(
#             subj,
#             n_trials,
#             data,
#             rl_alpha,
#             scaler,
#             a,
#             z,
#             t,
#             theta,
#             trial,
#             feedback,
#         )

#     return logp
# def make_logp_func(n_participants: int, n_trials_per_subj: jnp.ndarray) -> callable:
#     """Create a log likelihood function for the RLDM model with variable trials per subject."""

#     num_params = 6  # Adjust if your model uses a different number of parameters

#     def logp(data, rl_alpha, scaler, a, z, t, theta, participant_id, trial, feedback):
#         # Compute start and end indices for each subject's trials
#         trial_counts = n_trials_per_subj
#         trial_starts = jnp.concatenate([jnp.array([0]), jnp.cumsum(trial_counts)[:-1]])
#         trial_ends = jnp.cumsum(trial_counts)

#         def single_subj_logp(subj_idx):
#             start = trial_starts[subj_idx]
#             ntrials = trial_counts[subj_idx]
#             # Use dynamic_slice for JAX compatibility
#             data_subj = jax.lax.dynamic_slice(data, (start, 0), (ntrials, data.shape[1]))
#             rl_alpha_subj = jax.lax.dynamic_slice(rl_alpha, (start,), (ntrials,))
#             scaler_subj = jax.lax.dynamic_slice(scaler, (start,), (ntrials,))
#             a_subj = jax.lax.dynamic_slice(a, (start,), (ntrials,))
#             z_subj = jax.lax.dynamic_slice(z, (start,), (ntrials,))
#             t_subj = jax.lax.dynamic_slice(t, (start,), (ntrials,))
#             theta_subj = jax.lax.dynamic_slice(theta, (start,), (ntrials,))
#             trial_subj = jax.lax.dynamic_slice(trial, (start,), (ntrials,))
#             feedback_subj = jax.lax.dynamic_slice(feedback, (start,), (ntrials,))

#             return rlssm1_logp_inner_func(
#                 subj_idx,
#                 ntrials,
#                 data_subj,
#                 rl_alpha_subj,
#                 scaler_subj,
#                 a_subj,
#                 z_subj,
#                 t_subj,
#                 theta_subj,
#                 trial_subj,
#                 feedback_subj,
#             )

#         logps = jax.vmap(single_subj_logp)(jnp.arange(n_participants))
#         return logps.ravel()

#     return logp

# def make_rldm_logp_op(n_participants: int, n_trials_per_sub: list, n_params: int) -> callable:
#     """Create a pytensor Op for the likelihood function of RLDM model with variable trials per subject."""
#     logp = make_logp_func(n_participants, jnp.array(n_trials_per_sub))
#     vjp_logp = make_vjp_func(logp, params_only=False, n_params=n_params)

#     return make_jax_logp_ops(
#         logp=jax.jit(logp),
#         logp_vjp=jax.jit(vjp_logp),
#         logp_nojit=logp,
#         n_params=n_params,
#     )
def make_logp_func(n_participants: int, n_trials: int, n_states: int) -> Callable:
    """Create a log likelihood function for the RLDM model.

    Parameters
    ----------
    n_participants : int
        The number of participants in the dataset.
    n_trials : int
        The number of trials per participant.

    Returns
    -------
    callable
        A function that computes the log likelihood for the RLDM model.
    """

    # Ensure parameters are correctly extracted and passed to your custom function.
    def logp(data, *dist_params) -> jnp.ndarray:
        """Compute the log likelihood for the RLDM model.

        Parameters
        ----------
        dist_params
            A tuple containing the subject index, number of trials per subject,
            data, and model parameters. In this case, it is expected to be
            (rl_alpha, scaler, a, z, t, theta, trial, feedback).

        Returns
        -------
        jnp.ndarray
            The log likelihoods for each subject.
        """
        # Extract extra fields (adjust indices based on your model)
        participant_id = dist_params[num_params]
        trial = dist_params[num_params + 1]
        feedback = dist_params[num_params + 2]
        state1 = dist_params[num_params + 3]
        state2 = dist_params[num_params + 4]
        response2 = dist_params[num_params + 5]
        valid_upto = dist_params[num_params + 6] 
        




        subj = jnp.unique(participant_id, size=n_participants).astype(jnp.int32)

        # create parameter arrays to be passed to the likelihood function
        rl_alpha, scaler, a, z, t, theta, w = dist_params[:num_params]

        # pass the parameters and data to the likelihood function
        return vec_logp(
            subj,
            n_trials,
            data,
            rl_alpha,
            scaler,
            a,
            z,
            t,
            theta,
            w,
            trial,
            feedback,
            state1,
            state2,
            response2,
            valid_upto, 
            n_states
        )

    return logp


def make_rldm_logp_op(n_participants: int, n_trials: int, n_params: int, n_states: int) -> Callable:
# def make_rldm_logp_op(n_participants: int, n_trials_per_sub: list, n_params: int, max_trials: int) -> Callable:
    """Create a pytensor Op for the likelihood function of RLDM model.

    Parameters
    ----------
    n_participants : int
        The number of participants in the dataset.
    n_trials : int
        The number of trials per participant.

    Returns
    -------
    callable
        A function that computes the log likelihood for the RLDM model.
    """
    logp = make_logp_func(n_participants, n_trials, n_states)
    # logp = make_logp_func(n_participants, jnp.array(n_trials_per_sub), max_trials)
    vjp_logp = make_vjp_func(logp, params_only=False, n_params=n_params)

    return make_jax_logp_ops(
        logp=jax.jit(logp),
        logp_vjp=jax.jit(vjp_logp),
        logp_nojit=logp,
        n_params=n_params,
    )
__all__ = [
    "make_jax_logp_ops",
    "rlssm1_logp_inner_func",
    "make_rldm_logp_op",
    "make_logp_func",
    # add any other functions you want to use outside
]