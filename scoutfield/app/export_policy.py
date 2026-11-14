"""
ONNX export of the trained PPO policy, for running it without PyTorch.

Run (settings in ``configs/deployment.yaml`` under ``policy``):
    python experiments/08_export_policy.py
    python experiments/08_export_policy.py --checkpoint <path/to/ppo_seed0.zip>

What is exported
----------------
Only the actor: observation in, one score per action out. Stable-Baselines3 picks
the deterministic action of a discrete policy as the highest score, so
``ONNXPolicy.act`` takes the arg-max of the exported scores. The value head is not
needed to act and is left out.

The check that matters
----------------------
After export, the ONNX policy and the Stable-Baselines3 policy are run side by side
on observations from the real environment: PPO's own episodes on the 32x32 field,
observing the PlantDoc ``shift_eval`` pool at the shift-fitted reference, at several
temperatures. Every observation must produce the same action. One mismatch fails
the export.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

INPUT_NAME = "observation"
OUTPUT_NAME = "action_scores"


class ONNXPolicy:
    """An exported PPO policy behind the pilot's Agent contract, run by onnxruntime.

    ``act(obs, env) -> int``, as every planner in the sweep. No torch import.
    """

    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.is_file():
            raise FileNotFoundError(
                f"no exported policy at {self.onnx_path}; run "
                "python experiments/08_export_policy.py"
            )
        self.session = ort.InferenceSession(str(self.onnx_path),
                                            providers=["CPUExecutionProvider"])

    def reset(self, env) -> None:
        """No recurrent state; present to satisfy the Agent contract."""

    def scores(self, obs) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        return self.session.run([OUTPUT_NAME], {INPUT_NAME: x})[0][0]

    def act(self, obs, env=None) -> int:
        return int(np.argmax(self.scores(obs)))


def _actor(policy):
    """The actor of an SB3 ``ActorCriticPolicy`` as a plain module."""
    import torch

    class Actor(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, observation):
            features = self.policy.extract_features(observation,
                                                    self.policy.pi_features_extractor)
            latent = self.policy.mlp_extractor.forward_actor(features)
            return self.policy.action_net(latent)

    return Actor(policy).eval()


def resolve_policy(config, store=None, override: str | Path | None = None
                   ) -> tuple[Path, str]:
    """The policy checkpoint to export: from the store, or a local file."""
    from scoutfield.app.export_onnx import _shown

    cfg = config.section("policy")
    if override is not None:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"no policy at {path}. Pass the path of a local "
                                    f"{cfg['checkpoint_file']}, or omit --checkpoint.")
        return path, f"local file {_shown(path)}"
    from scoutfield.utils.artifacts import get_store

    store = store or get_store()
    path = store.pull(cfg["checkpoint_stage"], cfg["checkpoint_file"])
    return path, f"store {store.remote_path(cfg['checkpoint_stage'], cfg['checkpoint_file'])}"


def export_policy(checkpoint: str | Path, out_path: str | Path, opset: int) -> dict:
    """Write the actor of an SB3 PPO checkpoint to ``out_path``."""
    import torch
    from stable_baselines3 import PPO

    model = PPO.load(str(checkpoint), device="cpu")
    actor = _actor(model.policy)
    n_obs = int(np.prod(model.observation_space.shape))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor, torch.zeros(1, n_obs), str(out_path),
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_shapes={"observation": {0: torch.export.Dim.AUTO}},
        opset_version=int(opset), external_data=False, verbose=False,
    )
    return {"model": model, "actor": actor, "n_obs": n_obs,
            "n_actions": int(model.action_space.n)}


def verify_policy(model, actor, onnx_path, pool, reference_temperature: float,
                  settings, verify: dict, store=None) -> dict:
    """Run SB3 and ONNX on the same real observations; any differing action fails.

    The SB3 policy flies the episodes (so the observations are the ones it meets),
    and at every step both pick an action from the same observation.
    """
    import torch

    from scoutfield.app import demo

    onnx_policy = ONNXPolicy(onnx_path)
    need = int(verify["min_observations"])
    temperatures = [float(t) for t in verify["temperatures"]]
    seeds = [int(s) for s in verify["seeds"]]

    n, mismatches, max_diff, episodes = 0, 0, 0.0, 0
    while n < need:
        for temperature in temperatures:
            seed = seeds[episodes % len(seeds)] + 100 * (episodes // len(seeds))

            class _SB3:
                def act(self, obs, env=None):
                    nonlocal n, mismatches, max_diff
                    action, _ = model.predict(obs, deterministic=True)
                    with torch.no_grad():
                        ref = actor(torch.as_tensor(np.asarray(obs, np.float32))[None])[0]
                    scores = onnx_policy.scores(obs)
                    max_diff = max(max_diff, float(np.max(np.abs(ref.numpy() - scores))))
                    mismatches += int(int(action) != int(np.argmax(scores)))
                    n += 1
                    return int(action)

            demo.run_episode(_SB3(), settings, pool, reference_temperature,
                             temperature, seed)
            episodes += 1
    if mismatches or max_diff > float(verify["logit_atol"]):
        raise AssertionError(
            f"ONNX and Stable-Baselines3 policies disagree over {n} observations: "
            f"{mismatches} action(s) differ, max |score diff| = {max_diff:.3e} "
            f"(tolerance {float(verify['logit_atol']):.0e}). The export is rejected."
        )
    return {"n_observations": n, "episodes": episodes, "actions_identical": True,
            "max_abs_score_diff": max_diff, "atol": float(verify["logit_atol"]),
            "temperatures": temperatures,
            "verified_on": f"PPO episodes on the {settings.pool_split} pool at the "
                           f"{settings.reference_condition} reference"}


def run_policy_export(config_path: str | Path | None = None,
                      checkpoint: str | Path | None = None, push: bool = False,
                      store=None) -> dict:
    """Export, verify and record the policy; optionally file it in the store."""
    import onnx
    import onnxruntime as ort
    import stable_baselines3
    import torch

    from scoutfield.app import demo
    from scoutfield.app.export_onnx import git_state, load_deployment_config, work_dir
    from scoutfield.utils.artifacts import get_store, sha256_of

    config = load_deployment_config(config_path)
    cfg = config.section("policy")
    store = store or get_store()
    ckpt, source = resolve_policy(config, store=store, override=checkpoint)

    out_dir = work_dir(config, store.run_id)
    onnx_path = out_dir / cfg["policy_file"]
    exported = export_policy(ckpt, onnx_path, int(cfg["opset"]))

    settings = demo.load_settings()
    pool = demo.load_pool(settings.pool_split, store=store)
    reference = demo.fitted_temperature(settings.reference_condition, store=store)
    check = verify_policy(exported["model"], exported["actor"], onnx_path, pool, reference,
                          settings, cfg["verify"], store=store)
    print(f"exported {onnx_path.name}: {check['n_observations']} observations from "
          f"{check['episodes']} episodes, actions identical, max |score diff| "
          f"{check['max_abs_score_diff']:.2e}")

    record = {
        "run_id": store.run_id,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": {"source": source, "sha256": sha256_of(ckpt)},
        "onnx": {"file": onnx_path.name, "sha256": sha256_of(onnx_path),
                 "bytes": onnx_path.stat().st_size, "opset": int(cfg["opset"]),
                 "input": {"name": INPUT_NAME, "shape": ["batch", exported["n_obs"]],
                           "dtype": "float32"},
                 "output": {"name": OUTPUT_NAME, "shape": ["batch", exported["n_actions"]],
                            "dtype": "float32",
                            "meaning": "actor score per action; the deterministic "
                                       "action is the arg-max"}},
        "verification": check,
        **git_state(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "stable_baselines3": stable_baselines3.__version__,
                     "onnx": onnx.__version__, "onnxruntime": ort.__version__},
    }
    record_path = out_dir / cfg["record_file"]
    with record_path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    print(f"wrote {record_path}")

    if push:
        stage = str(config["stage"])
        recorded = store.stage_entry(cfg["checkpoint_stage"]).get("files", {}) \
            .get(cfg["checkpoint_file"], {}).get("sha256")
        if recorded != record["checkpoint"]["sha256"]:
            raise ValueError(f"refusing to push: the exported policy ({source}) is not the "
                             f"{cfg['checkpoint_file']} recorded for run {store.run_id}.")
        if record["git_dirty"]:
            raise ValueError("refusing to push an export made from uncommitted code; "
                             "commit first so the record names the code that made it.")
        store.push(stage, [onnx_path, record_path], "PPO policy exported to ONNX")
        store.record_inputs(stage, [(cfg["checkpoint_stage"], cfg["checkpoint_file"])])
        print(f"pushed {onnx_path.name} and {record_path.name} to {stage} "
              f"of run {store.run_id}")
    return record
