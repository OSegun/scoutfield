"""
PPO via Stable-Baselines3.

Run:
    python -m scoutfield.planners.ppo --config configs/ppo_field32.yaml

Why PPO rather than the pilot's REINFORCE
------------------------------------------
REINFORCE was chosen in the pilot for one reason only: it needs no deep learning
framework, so it runs on a bare NumPy kernel. That constraint bought
reproducibility at the cost of sample efficiency, and it is one of the three named
causes of the learned agent losing to a fixed sweep at every temperature.

PPO's clipped objective and value baseline cut gradient variance substantially,
which is what this problem needs: the reward is dominated by rare detection
events, so the raw policy gradient is extremely noisy. The other two named causes
are addressed by the environment change (a global belief map, and a field scale
where selectivity is worth something) rather than by the algorithm — worth being
explicit about, because attributing an improvement to PPO when it came from the
observation change would be wrong.

Honesty requirement
-------------------
The pilot reported plainly that its learned agent lost. If PPO also loses, that
gets reported the same way, with a stated reason. The comparison is only worth
anything if the answer was allowed to be negative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class PPOPlanner:
    """Trained SB3 policy exposed through the pilot's Agent contract.

    The wrapper is thin on purpose: the evaluation sweep should not know whether
    a planner is learned or hand-written, so that all five planners go through
    exactly the same evaluation path.
    """

    def __init__(self, model_path: str | Path, deterministic: bool = True):
        from stable_baselines3 import PPO

        self.model_path = Path(model_path)
        self.deterministic = deterministic
        # CPU is deliberate: evaluation is thousands of short episodes where
        # per-step transfer latency dominates, so GPU inference is slower here,
        # and it keeps the sweep runnable without a GPU quota.
        self.model = PPO.load(self.model_path, device="cpu")

    def reset(self, env) -> None:
        """No recurrent state to clear; present to satisfy the Agent contract."""

    def act(self, obs, env) -> int:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)


def _make_info_callback(episodes: list[dict] | None = None):
    """A callback recording the full ``info`` dict at every episode end.

    Episode return is not the quantity the project reports — detections-per-joule
    is — so logging only the return would leave the learning curve measuring
    something other than the result. ``episodes`` seeds the record with what a
    resumed run had already logged, so the curve covers the whole of training.

    Built inside a function rather than at module scope so that importing this
    module does not require SB3.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class _InfoDictCallback(BaseCallback):
        def __init__(self, verbose: int = 0):
            super().__init__(verbose)
            self.episodes: list[dict] = list(episodes or [])

        def _on_step(self) -> bool:
            for done, info in zip(self.locals.get("dones", []),
                                  self.locals.get("infos", [])):
                if done:
                    self.episodes.append({
                        "timestep": int(self.num_timesteps),
                        **{k: info[k] for k in (
                            "recall", "precision", "detections_per_joule",
                            "false_alarms", "coverage", "time_to_first_detection",
                        ) if k in info},
                    })
            return True

    return _InfoDictCallback()


def _make_push_callback(every: int, push):
    """A callback calling ``push()`` every ``every`` timesteps.

    ``push`` saves the resume state and the curve so far and uploads both, so a
    session killed mid-training loses at most ``every`` steps.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class _PushCallback(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self.last = None

        def _on_step(self) -> bool:
            if self.last is None:
                self.last = self.num_timesteps
            if self.num_timesteps - self.last >= every:
                self.last = self.num_timesteps
                push(self.model)
            return True

    return _PushCallback()


def _env_factory(config, base_seed: int, rank: int, temperature: float,
                 classifier=None, pool=None, reference_temperature: float = 1.0,
                 pool_split: str = ""):
    """A picklable zero-argument factory for one subprocess environment.

    Each rank gets a distinct seed, and its own classifier with its own generator.
    A single classifier pickled into every subprocess would carry the same
    generator state into each, so every rank would draw the same logits: the
    runs would be correlated and the confidence intervals silently too narrow.
    """
    from scoutfield.envs.field_env import make_field_env
    from scoutfield.envs.gym_wrapper import GymScoutEnv

    def _init():
        clf = classifier
        if clf is None:
            import numpy as np

            from scoutfield.perception.adapter import CNNClassifier

            clf = CNNClassifier(logit_pool=pool, temperature=temperature,
                                reference_temperature=reference_temperature,
                                rng=np.random.default_rng(base_seed + rank),
                                pool_split=pool_split)
        return GymScoutEnv(
            lambda seed=None: make_field_env(
                config,
                seed=base_seed + rank if seed is None else seed,
                temperature=temperature,
                classifier=clf,
            ),
            seed=base_seed + rank,
        )

    return _init


def train_ppo(config, seed: int | None = None, classifier=None,
              total_timesteps: int | None = None, store=None) -> Path:
    """Train to convergence and return the local path of the final policy.

    The policy, its resume state and its learning curve live in stage ``03_ppo``
    of the artefact store:

    * a seed whose final policy is already in the store is skipped;
    * a seed with resume state continues from it, so a killed session is resumed
      by running the notebook again;
    * resume state and the curve so far are pushed every ``ppo.checkpoint_freq``
      timesteps.

    Training runs at sweep T = 1 relative to ``env.reference_condition`` (the
    shift-fitted temperature), on the ``env.pool_split`` pool. Both, and the
    effective divisor, are written into the curve file.

    Convergence means the evaluation curve is flat over the last ~20% of training
    on every seed. Stopping at a fixed step count and calling it converged is the
    claim reviewers challenge first, so the flatness check is computed and
    recorded rather than asserted in prose.

    The reward weights are read from the config and never tuned here. Reward
    shaping consuming weeks is the project's top identified risk; if a shaping
    change becomes necessary it gets swept and reported, not adjusted until the
    numbers look better.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from scoutfield.utils.artifacts import get_store, git_commit
    from scoutfield.utils.paths import checkpoints_dir, results_dir

    store = store or get_store()
    ppo_cfg = config.section("ppo")
    env_cfg = config.section("env")
    seed = int(config["seed"] if seed is None else seed)
    total_timesteps = int(total_timesteps or ppo_cfg["total_timesteps"])
    n_envs = int(ppo_cfg["n_envs"])
    every = int(ppo_cfg["checkpoint_freq"])

    run_name = f"ppo_seed{seed}"
    final_name, resume_name = f"{run_name}.zip", f"{run_name}_resume.zip"
    curve_name = f"ppo_curve_seed{seed}.json"
    ckpt_dir = checkpoints_dir("ppo")
    final_path, resume_path = ckpt_dir / final_name, ckpt_dir / resume_name
    curve_path = results_dir() / curve_name

    if store.has("03_ppo", final_name):
        curve = store.pull_json("03_ppo", curve_name, required=False) or {}
        if curve.get("complete") and curve.get("total_timesteps") == total_timesteps:
            print(f"{final_name} is already complete in run {store.run_id}; skipping")
            return store.pull("03_ppo", final_name)

    pool_split = str(env_cfg["pool_split"])
    condition = str(env_cfg["reference_condition"])
    if classifier is None:
        from scoutfield.envs.field_env import _reference_temperature
        from scoutfield.perception.adapter import load_or_build_logit_pool

        # Resolved here, in the main process, before any subprocess starts:
        # SubprocVecEnv workers are daemonic and cannot run the DataLoader a pool
        # rebuild needs, and every rank needs the identical pool anyway.
        reference_temperature = _reference_temperature(condition, store=store)
        pool = load_or_build_logit_pool(pool_split, store=store)
        store.record_inputs("03_ppo")
    else:
        reference_temperature = float(getattr(classifier, "reference_temperature", 1.0))
        pool = None
    training_temperature = 1.0

    metadata = {
        "seed": seed,
        "total_timesteps": total_timesteps,
        "pool_split": pool_split,
        "reference_condition": condition,
        "reference_temperature": reference_temperature,
        "temperature": training_temperature,
        "effective_divisor": reference_temperature * training_temperature,
        "git_commit": git_commit(),
    }

    factories = [_env_factory(config, seed * 1000, r, training_temperature, classifier,
                              pool, reference_temperature, pool_split)
                 for r in range(n_envs)]
    # SubprocVecEnv only pays for itself with several environments, and it needs a
    # __main__ guard on Windows; a single env is cheaper in-process.
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    venv = vec_cls(factories)

    def write_curve(episodes, complete: bool, resumed_from: int | None) -> None:
        with curve_path.open("w", encoding="utf-8") as fh:
            json.dump({**metadata, "complete": complete,
                       "resumed_from_timestep": resumed_from,
                       "episodes": episodes,
                       "converged": _is_flat(episodes) if complete else None}, fh, indent=2)

    try:
        resumed_from, episodes = None, []
        if store.has("03_ppo", resume_name):
            # A killed session resumes rather than restarting.
            model = PPO.load(store.pull("03_ppo", resume_name), env=venv, device="auto")
            partial = store.pull_json("03_ppo", curve_name, required=False) or {}
            episodes = list(partial.get("episodes", []))
            resumed_from = int(model.num_timesteps)
            print(f"resumed {run_name} at timestep {resumed_from}")
        else:
            model = PPO(
                ppo_cfg.get("policy", "MlpPolicy"),
                venv,
                learning_rate=float(ppo_cfg["learning_rate"]),
                n_steps=int(ppo_cfg["n_steps"]),
                batch_size=int(ppo_cfg["batch_size"]),
                n_epochs=int(ppo_cfg["n_epochs"]),
                gamma=float(ppo_cfg["gamma"]),
                gae_lambda=float(ppo_cfg["gae_lambda"]),
                clip_range=float(ppo_cfg["clip_range"]),
                ent_coef=float(ppo_cfg["ent_coef"]),
                vf_coef=float(ppo_cfg["vf_coef"]),
                max_grad_norm=float(ppo_cfg["max_grad_norm"]),
                policy_kwargs={"net_arch": list(ppo_cfg["net_arch"])},
                seed=seed,
                verbose=1,
            )

        info_cb = _make_info_callback(episodes)

        def push_resume(m) -> None:
            m.save(resume_path)
            write_curve(info_cb.episodes, complete=False, resumed_from=resumed_from)
            store.push("03_ppo", [resume_path, curve_path],
                       f"{run_name} at timestep {m.num_timesteps}")

        remaining = total_timesteps - int(model.num_timesteps)
        if remaining > 0:
            # On a resumed model SB3 adds `total_timesteps` to the steps already
            # taken, so only the remainder is passed.
            model.learn(total_timesteps=remaining,
                        callback=[info_cb, _make_push_callback(every, push_resume)],
                        reset_num_timesteps=resumed_from is None)
        model.save(final_path)
        model.save(resume_path)
    finally:
        venv.close()

    write_curve(info_cb.episodes, complete=True, resumed_from=resumed_from)
    store.push("03_ppo", [final_path, resume_path, curve_path], f"{run_name} complete")
    print(f"pushed {final_name} and {curve_name} to run {store.run_id}")
    return final_path


def _is_flat(episodes: list[dict], metric: str = "detections_per_joule",
             tail: float = 0.2, tolerance: float = 0.05) -> bool | None:
    """Whether the metric is flat over the final ``tail`` of training.

    Compares the mean over the last fifth against the fifth before it. Returns
    None when there is not enough data to judge, rather than a default answer —
    reporting "converged" on three episodes would be worse than reporting nothing.
    """
    values = [e[metric] for e in episodes if metric in e]
    if len(values) < 50:
        return None
    cut = int(len(values) * (1.0 - tail))
    prev_cut = int(len(values) * (1.0 - 2 * tail))
    last = sum(values[cut:]) / max(len(values[cut:]), 1)
    prior = sum(values[prev_cut:cut]) / max(len(values[prev_cut:cut]), 1)
    if prior == 0:
        return None
    return abs(last - prior) / abs(prior) < tolerance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ppo_field32.yaml")
    parser.add_argument("--seed", type=int, default=None, help="overrides config seed")
    args = parser.parse_args()

    from scoutfield.config import load_config
    from scoutfield.utils.seeding import seed_everything

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["seed"]
    seed_everything(seed)
    print("checkpoint:", train_ppo(config, seed=seed))


if __name__ == "__main__":
    main()
