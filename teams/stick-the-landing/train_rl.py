#!/usr/bin/env python3
"""Train an RL agent for the Catch the Boat challenge.

Usage (local, baseline sim):
    python teams/stick-the-landing/train_rl.py --scenario easy --timesteps 200000

Usage (Docker, reference sim — recommended):
    CHALLENGE_IMAGE=challenge-eval:full ./docker/run-local.sh \\
        python teams/stick-the-landing/train_rl.py \\
        --scenario easy --timesteps 500000 --use-reference-sim

Curriculum training (recommended):
    python teams/stick-the-landing/train_rl.py --curriculum --timesteps 300000
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Make the repo root importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def make_env(
    scenario: str,
    use_reference_sim: bool = False,
    seed: Optional[int] = None,
    rank: int = 0,
):
    """Factory function for creating training environments.

    Returns a callable (thunk) for SubprocVecEnv compatibility.
    """
    def _init():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rl_env",
            str(Path(__file__).resolve().parent / "rl_env.py"),
        )
        rl_env_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rl_env_mod)
        BoatLandingGymWrapper = rl_env_mod.BoatLandingGymWrapper

        drone_sim = None
        if use_reference_sim:
            try:
                from boat_landing.reference_sim import make_drone_sim
                drone_sim = make_drone_sim(
                    str(REPO_ROOT / "drones" / "vtol.yaml")
                )
            except ImportError:
                print(
                    "[train_rl] WARNING: reference sim not available. "
                    "Using baseline sim.",
                    file=sys.stderr,
                )

        env = BoatLandingGymWrapper(
            scenario=scenario,
            drone_sim=drone_sim,
            seed=(seed + rank) if seed is not None else None,
        )
        return env

    return _init


def train(
    scenario: str = "easy",
    total_timesteps: int = 200_000,
    n_envs: int = 4,
    use_reference_sim: bool = False,
    resume: bool = False,
    seed: int = 42,
    learning_rate: float = 3e-4,
    log_dir: Optional[str] = None,
):
    """Train PPO on the specified scenario."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import (
        SubprocVecEnv,
        DummyVecEnv,
        VecMonitor,
    )
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
    )

    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "landing_policy"

    if log_dir is None:
        log_dir = str(
            Path(__file__).resolve().parent / "tb_logs" / scenario
        )

    print(f"[train_rl] Scenario: {scenario}", file=sys.stderr)
    print(f"[train_rl] Timesteps: {total_timesteps:,}", file=sys.stderr)
    print(f"[train_rl] Envs: {n_envs}", file=sys.stderr)
    print(
        f"[train_rl] Reference sim: {use_reference_sim}", file=sys.stderr
    )
    print(f"[train_rl] Model path: {model_path}", file=sys.stderr)

    # Create vectorized training envs
    # Use DummyVecEnv for single env (easier debugging),
    # SubprocVecEnv for multiple envs (parallelism)
    env_fns = [
        make_env(scenario, use_reference_sim, seed, rank=i)
        for i in range(n_envs)
    ]
    if n_envs == 1:
        train_env = DummyVecEnv(env_fns)
    else:
        train_env = SubprocVecEnv(env_fns)
    train_env = VecMonitor(train_env)

    # Create eval env (single, deterministic)
    eval_env = DummyVecEnv(
        [make_env(scenario, use_reference_sim, seed=seed + 1000)]
    )
    eval_env = VecMonitor(eval_env)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1_000),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="landing",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(model_dir / "eval_logs"),
        eval_freq=max(20_000 // n_envs, 1_000),
        n_eval_episodes=5,
        deterministic=True,
    )

    # Load or create model
    if resume and (model_path.with_suffix(".zip")).exists():
        print(
            f"[train_rl] Resuming from {model_path}.zip", file=sys.stderr
        )
        model = PPO.load(
            str(model_path),
            env=train_env,
            tensorboard_log=log_dir,
        )
        model.learning_rate = learning_rate
    else:
        print("[train_rl] Creating new model", file=sys.stderr)
        model = PPO(
            "MlpPolicy",
            train_env,
            policy_kwargs=dict(
                net_arch=[256, 256],  # 2 hidden layers, ~130K params
                activation_fn=__import__("torch").nn.Tanh,
            ),
            learning_rate=learning_rate,
            n_steps=2048,       # steps per rollout per env
            batch_size=64,
            n_epochs=10,
            gamma=0.99,         # discount factor
            gae_lambda=0.95,    # GAE lambda
            clip_range=0.2,     # PPO clip
            ent_coef=0.01,      # entropy bonus (encourages exploration)
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=seed,
            verbose=1,
            tensorboard_log=log_dir,
        )

    # Train
    print(
        f"\n[train_rl] Starting training for {total_timesteps:,} steps...",
        file=sys.stderr,
    )
    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )
    elapsed = time.time() - t0
    print(
        f"\n[train_rl] Training complete in {elapsed:.1f}s "
        f"({elapsed / 60:.1f} min)",
        file=sys.stderr,
    )

    # Save final model
    model.save(str(model_path))
    print(f"[train_rl] Model saved to {model_path}.zip", file=sys.stderr)

    # Cleanup
    train_env.close()
    eval_env.close()

    return model


def train_curriculum(
    timesteps_per_stage: int = 300_000,
    n_envs: int = 4,
    use_reference_sim: bool = False,
    seed: int = 42,
):
    """Curriculum training: EASY → MEDIUM → HARD.

    Each stage fine-tunes the model from the previous stage with a
    decaying learning rate.
    """
    scenarios = ["easy", "medium", "hard"]
    learning_rates = [3e-4, 1e-4, 5e-5]  # decay across stages

    for i, (scenario, lr) in enumerate(zip(scenarios, learning_rates)):
        print(
            f"\n{'='*60}\n"
            f"[CURRICULUM] Stage {i+1}/3: {scenario.upper()} "
            f"(lr={lr}, steps={timesteps_per_stage:,})\n"
            f"{'='*60}",
            file=sys.stderr,
        )
        train(
            scenario=scenario,
            total_timesteps=timesteps_per_stage,
            n_envs=n_envs,
            use_reference_sim=use_reference_sim,
            resume=(i > 0),  # resume from previous stage
            seed=seed,
            learning_rate=lr,
        )

    print(
        "\n[CURRICULUM] All stages complete! Model saved.",
        file=sys.stderr,
    )


def smoke_test(
    scenario: str = "easy",
    timesteps: int = 1000,
    n_envs: int = 1,
    seed: int = 42,
) -> bool:
    """1000-step RL smoke test for the rl-integration plumbing.

    Instantiates the gym wrapper (which forces the GAP 1/2/3 fixes to
    load), trains PPO for a small number of timesteps, asserts the model
    saves without error. Designed to validate the integration end-to-end
    without burning GPU time.

    Returns True on success, raises on failure.
    """
    print(
        f"[RL-int] [train_rl.smoke] scenario={scenario} timesteps={timesteps} "
        f"n_envs={n_envs} seed={seed}",
        file=sys.stderr,
    )
    t0 = time.time()
    train(
        scenario=scenario,
        total_timesteps=timesteps,
        n_envs=n_envs,
        use_reference_sim=False,
        resume=False,
        seed=seed,
    )
    # Verify model.save() succeeded by checking the file exists.
    model_path = (
        Path(__file__).resolve().parent / "models" / "landing_policy.zip"
    )
    assert model_path.exists(), (
        f"[RL-int] smoke FAILED: model not saved at {model_path}"
    )
    elapsed = time.time() - t0
    print(
        f"[RL-int] [train_rl.smoke] PASS in {elapsed:.1f}s "
        f"(model at {model_path}, "
        f"{model_path.stat().st_size / 1024:.1f} KB)",
        file=sys.stderr,
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agent for Catch the Boat"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the 1000-step rl-integration smoke test and exit.",
    )
    parser.add_argument(
        "--scenario",
        default="easy",
        help="Scenario to train on (easy/medium/hard). "
        "Ignored if --curriculum is set.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps (per stage if curriculum).",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel training environments.",
    )
    parser.add_argument(
        "--use-reference-sim",
        action="store_true",
        help="Use the organizer reference simulator (Docker only).",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Train with curriculum: EASY → MEDIUM → HARD.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing model checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    args = parser.parse_args()

    if args.smoke:
        smoke_timesteps = args.timesteps if args.timesteps != 200_000 else 1000
        smoke_test(
            scenario=args.scenario,
            timesteps=smoke_timesteps,
            n_envs=1,
            seed=args.seed,
        )
        return

    if args.curriculum:
        train_curriculum(
            timesteps_per_stage=args.timesteps,
            n_envs=args.n_envs,
            use_reference_sim=args.use_reference_sim,
            seed=args.seed,
        )
    else:
        train(
            scenario=args.scenario,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            use_reference_sim=args.use_reference_sim,
            resume=args.resume,
            seed=args.seed,
            learning_rate=args.lr,
        )


if __name__ == "__main__":
    main()
