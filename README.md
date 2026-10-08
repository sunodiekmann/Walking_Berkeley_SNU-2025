# Walking Berkeley

Reinforcement-learning locomotion for the **Berkeley Humanoid Lite** (BHL), trained in MuJoCo/MJX with Brax PPO and checked sim-to-sim in plain MuJoCo with an exported ONNX policy.

Built from [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground), which was inspired by [kscalelabs/mujoco_playground](https://github.com/kscalelabs/mujoco_playground).

## Installation

Install [uv](https://docs.astral.sh/uv/), then sync the environment from the repo root:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## Training

```bash
uv run playground/berkeley_humanoid_lite/runner.py --env joystick --task flat_terrain
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--env` | `joystick` | `joystick` (velocity-command walking) or `standing` |
| `--task` | `flat_terrain` | Scene; see `constants.task_to_xml` |
| `--num_timesteps` | `300000000` | Total PPO environment steps |
| `--output_dir` | `checkpoints` | Where checkpoints and `.onnx` exports are written (relative to the current directory) |
| `--restore_checkpoint_path` | none | Resume from a checkpoint |

Each evaluation step writes a checkpoint and an ONNX export named `<date>_<step>` to `--output_dir`.

The imitation reward uses `playground/berkeley_humanoid_lite/data/polynomial_coefficients.pkl`. Turn it on or off with `USE_IMITATION_REWARD` at the top of `joystick.py` or `standing.py`.

TensorBoard:

```bash
uv run tensorboard --logdir=checkpoints
```

## Inference (MuJoCo sim-to-sim)

```bash
uv run playground/berkeley_humanoid_lite/mujoco_infer.py -o policies/Final_Policy.onnx
```

Add `--standing` for a standing policy. The scene XML and reference-motion data have default paths; override them with `--model_path` and `--reference_data`. Press Ctrl+C to save the observations to `mujoco_saved_obs.pkl`, then plot them with `uv run playground/common/plot_saved_obs.py`.

To export a checkpoint to ONNX after training:

```bash
uv run playground/common/sep_export.py --checkpoint_dir checkpoints/<run> --output_path policies/<name>.onnx
```

## Project structure

```
.
├── pyproject.toml
├── policies/                      # trained .onnx policies (gitignored)
└── playground/
    ├── common/                    # shared training infrastructure, not robot-specific
    │   ├── runner.py              # BaseRunner: PPO training loop, checkpointing, ONNX export
    │   ├── brax_train.py          # vendored Brax PPO trainer
    │   ├── export_onnx.py         # Brax policy → ONNX (via TF)
    │   ├── sep_export.py          # CLI: export an existing checkpoint to ONNX
    │   ├── onnx_infer.py          # ONNX runtime wrapper
    │   ├── poly_reference_motion.py        # reference motion (JAX, used in training)
    │   ├── poly_reference_motion_numpy.py  # reference motion (NumPy, used in inference)
    │   ├── rewards.py / rewards_numpy.py   # generic locomotion rewards
    │   ├── randomize.py           # domain randomization
    │   ├── plot_saved_obs.py      # plot observations saved by mujoco_infer
    │   └── utils.py
    └── berkeley_humanoid_lite/
        ├── constants.py           # paths, joint/site/sensor names
        ├── base.py                # BHL MJX base env
        ├── joystick.py            # walking env (velocity commands)
        ├── standing.py            # standing env
        ├── custom_rewards.py / custom_rewards_numpy.py
        ├── runner.py              # training entry point
        ├── mujoco_infer.py        # sim-to-sim entry point
        ├── mujoco_infer_base.py
        ├── data/                  # polynomial reference motion (.pkl + .xlsx dump)
        ├── xmls/                  # MJCF: robot.xml, scene_flat_terrain.xml, meshes
        └── tools/                 # development utilities (see below)
```

## Tools

Run these from the repo root:

| Command | Purpose |
| --- | --- |
| `uv run playground/berkeley_humanoid_lite/tools/view_model.py` | Open the scene in the MuJoCo viewer at the `home` keyframe |
| `uv run playground/berkeley_humanoid_lite/tools/ref_motion_viewer.py [--command dx dy dtheta] [-joystick]` | Play back the reference motion |
| `uv run playground/berkeley_humanoid_lite/tools/check_checkpoint.py checkpoints/<run>` | Check a checkpoint for NaN/Inf values |
| `uv run playground/berkeley_humanoid_lite/tools/debug_overflow.py` | Build the joystick env with overflow warnings raised as errors |
| `uv run playground/berkeley_humanoid_lite/tools/ref_motion_to_excel.py` | Dump `data/polynomial_coefficients.pkl` to `.xlsx` |
