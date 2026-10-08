#!/usr/bin/env python
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import numpy as np
from orbax import checkpoint as ocp
from playground.common.export_onnx import export_onnx


def load_params_from_checkpoint(ckpt_dir: str):
    ckpt_dir = os.path.abspath(ckpt_dir)
    print(f"[offline_export] Restoring checkpoint from: {ckpt_dir}")

    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(ckpt_dir)

    params = restored
    print(f"[offline_export] Restored params type: {type(params)}")
    if not isinstance(params, (tuple, list)):
        raise RuntimeError(
            f"Expected params to be a tuple/list (normalization, train_state, ...), "
            f"got {type(params)} instead."
        )
    return params


def extract_policy_params(train_state_like):
    policy = getattr(train_state_like, "policy", train_state_like)
    return policy.get("params") if isinstance(policy, dict) else None


def infer_obs_size(params) -> int:
    norm_params = params[0]

    # norm_params is now a dict like {"mean": {...}, "std": {...}}
    if hasattr(norm_params, "mean"):
        mean_dict = norm_params.mean
    elif isinstance(norm_params, dict) and "mean" in norm_params:
        mean_dict = norm_params["mean"]
    else:
        raise RuntimeError(
            f"Cannot find 'mean' in normalization params (type={type(norm_params)}). "
        )

    if isinstance(mean_dict, dict):
        if "state" not in mean_dict:
            raise RuntimeError(
                f"'mean' dict does not contain key 'state'. Available keys: {list(mean_dict.keys())}"
            )
        mean_state = np.array(mean_dict["state"])
    else:
        mean_state = np.array(mean_dict)

    obs_size = int(mean_state.shape[-1])
    print(f"[offline_export] Inferred obs_size = {obs_size}")
    return obs_size


def infer_action_size(params) -> int:
    train_state_like = params[1]
    jax_params = extract_policy_params(train_state_like)
    if jax_params is None:
        raise RuntimeError(
            "Could not extract policy params from params[1]. "
            "Check the structure of your checkpoint."
        )

    hidden_layer_names = [n for n in jax_params.keys() if n.startswith("hidden_")]
    if not hidden_layer_names:
        raise RuntimeError(
            "No 'hidden_X' layers found in policy params. "
            "Got keys: " + ", ".join(jax_params.keys())
        )

    hidden_layer_names.sort(key=lambda n: int(n.split("_")[1]))
    last_name = hidden_layer_names[-1]
    last_kernel = np.array(jax_params[last_name]["kernel"])
    out_dim = int(last_kernel.shape[1])
    if out_dim % 2 != 0:
        raise RuntimeError(
            f"Last layer '{last_name}' out_dim={out_dim} is not even; "
            "cannot infer action_size = out_dim / 2."
        )
    act_size = out_dim // 2
    print(
        f"[offline_export] {last_name}: kernel {last_kernel.shape} "
        f"-> param_size={out_dim}, inferred action_size={act_size}"
    )
    return act_size


class DummyNetFactory:
    def __init__(self, policy_hidden_layer_sizes):
        self.policy_hidden_layer_sizes = list(policy_hidden_layer_sizes)


class DummyPPOParams:
    def __init__(self, policy_hidden_layer_sizes):
        self.network_factory = DummyNetFactory(policy_hidden_layer_sizes)


def adapt_params_for_export(params):
    """Make params[0] look like the old object with .mean/.std for export_onnx."""
    norm_params = params[0]

    # If it's already an object with .mean, just return as-is.
    if hasattr(norm_params, "mean") and hasattr(norm_params, "std"):
        return params

    if not (isinstance(norm_params, dict) and "mean" in norm_params and "std" in norm_params):
        raise RuntimeError(
            f"Cannot adapt normalization params for export; got type={type(norm_params)}"
        )

    class NormWrapper:
        pass

    norm_wrapper = NormWrapper()
    norm_wrapper.mean = norm_params["mean"]
    norm_wrapper.std = norm_params["std"]

    # Replace only index 0, keep everything else unchanged
    if isinstance(params, list):
        new_params = list(params)
    else:
        new_params = list(params)
    new_params[0] = norm_wrapper
    return new_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="ABSOLUTE path to the checkpoint directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="policy.onnx",
        help="Where to save the exported ONNX model.",
    )
    parser.add_argument(
        "--policy_hidden",
        type=str,
        default="512,256,128",
        help="Comma-separated policy hidden sizes; must match training.",
    )
    args = parser.parse_args()

    ckpt_dir = os.path.abspath(args.checkpoint_dir)
    if not os.path.isdir(ckpt_dir):
        raise SystemExit(f"Checkpoint dir does not exist: {ckpt_dir}")

    # 1) restore params
    params = load_params_from_checkpoint(ckpt_dir)

    # 2) infer obs_size and act_size
    obs_size = infer_obs_size(params)
    act_size = infer_action_size(params)

    # 3) parse known hidden sizes
    hidden_sizes = [int(x) for x in args.policy_hidden.split(",") if x.strip()]
    print(f"[offline_export] Using policy_hidden_layer_sizes = {hidden_sizes}")

    # 4) build dummy ppo_params
    ppo_params = DummyPPOParams(hidden_sizes)

    # 5) adapt params[0] so export_onnx sees .mean / .std like before
    export_params = adapt_params_for_export(params)

    # 6) call your ORIGINAL export_onnx
    print(
        f"[offline_export] Calling export_onnx(params, act_size={act_size}, "
        f"obs_size={obs_size}, hidden_sizes={hidden_sizes})"
    )
    export_onnx(
        params=export_params,
        act_size=act_size,
        ppo_params=ppo_params,
        obs_size=obs_size,
        output_path=args.output_path,
    )
    print(f"[offline_export] Export done. ONNX saved to: {args.output_path}")


if __name__ == "__main__":
    main()
