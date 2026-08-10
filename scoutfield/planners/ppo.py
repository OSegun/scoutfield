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


def _make_info_callback():
    """A callback recording the full ``info`` dict at every episode end.

    Episode return is not the quantity the paper reports — detections-per-joule
    is — so logging only the return would leave the learning curve measuring
    something other than the result.

    Built inside a function rather than at module scope so that importing this
    module does not require SB3.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class _InfoDictCallback(BaseCallback):
        def __init__(self, verbose: int = 0):
            super().__init__(verbose)
            self.episodes: list[dict] = []

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


def _env_factory(config, base_seed: int, rank: int, temperature: float, classifier=None):
    """A picklable zero-argument factory for one subprocess environment.

    Each rank gets a distinct seed. Sharing one env instance across subprocesses
    would correlate the runs and silently narrow the confidence intervals.
    """
    from scoutfield.envs.field_env import make_field_env
    from scoutfield.envs.gym_wrapper import GymScoutEnv

    def _init():
        return GymScoutEnv(
            lambda seed=None: make_field_env(
                config,
                seed=base_seed + rank if seed is None else seed,
                temperature=temperature,
                classifier=classifier,
            ),
            seed=base_seed + rank,
        )

    return _init


def train_ppo(config, seed: int | None = None, classifier=None,
              total_timesteps: int | None = None) -> Path:
    """Train to convergence and return the checkpoint path.

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
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from scoutfield.utils.paths import checkpoints_dir, results_dir

    ppo_cfg = config.section("ppo")
    seed = int(config["seed"] if seed is None else seed)
    total_timesteps = int(total_timesteps or ppo_cfg["total_timesteps"])
    n_envs = int(ppo_cfg["n_envs"])

    ckpt_dir = checkpoints_dir("ppo")
    run_name = f"ppo_seed{seed}"
    final_path = ckpt_dir / f"{run_name}.zip"
    resume_path = ckpt_dir / f"{run_name}_resume.zip"

    factories = [_env_factory(config, seed * 1000, r, 1.0, classifier)
                 for r in range(n_envs)]
    # SubprocVecEnv only pays for itself with several environments, and it needs a
    # __main__ guard on Windows; a single env is cheaper in-process.
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    venv = vec_cls(factories)

    try:
        info_cb = _make_info_callback()
        checkpoint_cb = CheckpointCallback(
            save_freq=max(int(ppo_cfg["checkpoint_freq"]) // n_envs, 1),
            save_path=str(ckpt_dir),
            name_prefix=run_name,
        )

        if resume_path.exists():
            # A killed Kaggle session resumes rather than restarting.
            model = PPO.load(resume_path, env=venv, device="auto")
            print(f"resumed from {resume_path}")
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

        model.learn(total_timesteps=total_timesteps,
                    callback=[checkpoint_cb, info_cb],
                    reset_num_timesteps=not resume_path.exists())
        model.save(final_path)
        model.save(resume_path)
    finally:
        venv.close()

    curve_path = results_dir() / f"ppo_curve_seed{seed}.json"
    with curve_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "seed": seed,
            "total_timesteps": total_timesteps,
            "episodes": info_cb.episodes,
            "converged": _is_flat(info_cb.episodes),
        }, fh, indent=2)
    print(f"wrote {curve_path}")
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
