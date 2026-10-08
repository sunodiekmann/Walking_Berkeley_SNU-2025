# Handover: finishing BHL walking training

Status as of 2026-10-08. Goal: get the `joystick` walking policy for the Berkeley Humanoid Lite training well, starting by adding DeepMind-style **reference state initialization (RSI)**.

## Desktop setup

```bash
git clone git@github.com:sunodiekmann/Walking_Berkeley.git && cd Walking_Berkeley
uv sync
```

Untracked files you need to copy over yourself if you want them (they are gitignored):
- `policies/`: `Final_Policy.onnx` (latest), `BEST_WALK_ONNX*.onnx`, and 202 timestamped exports in `policies/exports/`
- `playground/checkpoints/`: Orbax checkpoints from Nov–Dec 2025 (~400 MB)

To check that the env builds without starting a full training run:

```bash
uv run python -c "import jax; from playground.berkeley_humanoid_lite import joystick; \
env = joystick.Joystick(task='flat_terrain'); s = env.reset(jax.random.PRNGKey(0)); \
print(jax.jit(env.step)(s, jax.numpy.zeros(env.action_size)).reward)"
```

Today this **fails** because of the bugs listed below. `standing.Standing` builds and steps fine.

## What works

- `standing` env: reset and step work.
- `mujoco_infer.py`: loads the scene and the reference motion through the NumPy loader.
- `common/sep_export.py` and `tools/check_checkpoint.py`: tested on a real checkpoint.

## Blocking bugs in the `joystick` env (fix these first)

The last commit before the cleanup was made partway through refactoring the reference-motion format. The NumPy (inference) side was updated, but the JAX (training) side was not.

1. **The JAX reference-motion loader doesn't accept the new `.pkl` format.**
   `common/poly_reference_motion.py` (`process`, around line 74) reads `data[name]["startend_double_support_ratio"]`, which the current `data/polynomial_coefficients.pkl` doesn't have. It fails with `KeyError`.
   Fix: port the format handling from `common/poly_reference_motion_numpy.py`, which already reads the new format (`nb_steps_in_period`, `degree`, `y_dim`, coefficients stored low→high), into the JAX version. Keep the JAX version traceable (`jp.polyval` / indexing by `imitation_i`).

2. **`reward_imitation` call doesn't match its definition.**
   `joystick.py` (`_get_reward`, around line 1285) passes 10 arguments and unpacks `imi, dbg`. `custom_rewards.py:158` takes 8 arguments and returns a single scalar. There is also an unused `reward_imitationnew` at `custom_rewards.py:4`. Pick one signature, delete the other, and make the call site match.

3. **Reward slices don't match the reference layout.** This one doesn't crash, so it's easy to miss.
   The `.pkl` frame has `y_dim = 60` with this layout:

   | Slice | Content |
   | --- | --- |
   | 0:20 | joints_pos (20) |
   | 20:40 | joints_vel (20) |
   | 40:43 / 43:46 | left / right toe pos |
   | 46:49 / 49:52 | left / right toe vel |
   | 52:54 | foot_contacts |
   | 54:57 | base_linear_vel |
   | 57:60 | base_angular_vel |

   `reward_imitation` (around `custom_rewards.py:185`) hardcodes a 12-joint layout (joint pos 0:12, vel 12:24, linvel 26:29, contacts 24:…), so it compares the wrong values. The 20 joints are probably 12 leg + 8 arm (the 8 arm motors are commented out in `xmls/robot.xml`), but the `.pkl` doesn't store joint names. **Confirm the order of the 20 joints** against the reference-motion generator before slicing. Then build an index map from the reference joints to the 12 actuated leg joints, in actuator order (`constants.JOINTS_ORDER_NO_HEAD`, ankle→hip, left then right). Ideally, compute the slices from `y_layout` instead of hardcoding them.

Verify the fixes by running the snippet above with `USE_IMITATION_REWARD = True` and with it set to `False`. Both should reset and step, and the imitation reward should be roughly at its maximum when the robot is driven exactly along the reference.

## Next feature: reference state initialization (RSI)

Currently, `Joystick.reset` (`joystick.py:539`) always starts from the `home` keyframe with small noise, and sets `imitation_i = 0`. As a result, the policy only experiences the start of each gait cycle at the beginning of an episode, and the imitation reward gives little signal in early training.

DeepMind-style RSI: in `reset`, after sampling `cmd`,

1. Sample a phase `i ~ U{0, nb_steps_in_period-1}` and set `info["imitation_i"] = i` and `info["imitation_phase"] = [cos, sin](2π·i/N)`. `step()` already increments these.
2. Get `ref = PRM.get_reference_motion(cmd[0], cmd[1], cmd[2], i)`.
3. Set the actuated joints' `qpos` to `ref` joints_pos (through the index map from bug 3), and their `qvel` to `ref` joints_vel.
4. Set the base linear and angular velocity from `ref`. It's unclear which frame they are in: `y_layout` calls them `base_*_vel`, but `frame_offsets` calls them `world_*_vel`. Check the generator. If they're in the body frame, rotate them by the sampled yaw before writing them to the free-joint `qvel`.
5. The reference has **no root height or orientation** in `y`. Keep the keyframe height and orientation, then correct the height so the lowest foot touches the ground: run `mjx.forward`, read the foot site z, and shift the base z. Otherwise the robot spawns floating or partly underground.
6. Set `ctrl` / `motor_targets` / `last_act` consistently with the new joint `qpos`, so the first action isn't a jump.
7. Keep a fraction of resets at the plain `home` pose (for example 20%) so the policy can still start from standstill. Make this fraction configurable (`config.rsi_prob`).

Everything must stay `jax.jit`- and `vmap`-safe: use `jax.random` and `jp.where`, not Python branching on traced values.

## Known open questions

- Where does `data/polynomial_coefficients.pkl` come from, and with which generator settings? The `.pkl` has 90 (dx, dy, dθ) entries with period 1.04 s at 50 fps. Regenerate it with 12 leg joints only if that's simpler than mapping.
- Reward weights in `default_config()` (`joystick.py:36`) were tuned while the imitation reward was broken. Re-tune them after the fixes, starting with `imitation = 7.0`.
- `standing.py` has `USE_IMITATION_REWARD = False` and is otherwise independent of these bugs.

## Repo layout

See `README.md`. In short: `playground/common/` holds the robot-agnostic training infrastructure, `playground/berkeley_humanoid_lite/` holds the envs, rewards, MJCF and entry points, and `playground/berkeley_humanoid_lite/tools/` holds dev scripts.
