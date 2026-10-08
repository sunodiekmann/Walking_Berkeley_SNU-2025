# Check that a training checkpoint contains only finite values.
import argparse
import os

import numpy as np, jax
from orbax.checkpoint import PyTreeCheckpointer

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", help="Path to a checkpoint directory, e.g. checkpoints/2025_11_02_044758_151388160")
ckpt = os.path.abspath(parser.parse_args().checkpoint)
restored = PyTreeCheckpointer().restore(ckpt)

def tree_all_finite(t):
    flat = jax.tree_util.tree_leaves(t)
    return all(np.isfinite(np.asarray(x)).all() for x in flat)

stats, policy, value = restored
print("stats finite:", tree_all_finite(stats))
print("policy finite:", tree_all_finite(policy))
print("value finite:", tree_all_finite(value))

# Also print ranges for the normalizer used at inference:
mean = np.asarray(stats["mean"]["state"])
std  = np.asarray(stats["std"]["state"])
print("mean/state: min/max", float(mean.min()), float(mean.max()))
print("std/state : min/max", float(std.min()),  float(std.max()))
print("any tiny std? ->", bool((std < 1e-8).any()))
