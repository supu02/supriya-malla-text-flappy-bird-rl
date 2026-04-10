from __future__ import annotations

import random
from collections.abc import Callable, Iterable

import gymnasium as gym
import numpy as np
import pandas as pd
import text_flappy_bird_gym

from rl_assignment.config import EnvConfig, TrainingConfig


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_env(env_config: EnvConfig):
    return gym.make("TextFlappyBird-v0", **env_config.as_kwargs())


def evaluate_agent(
    agent,
    env_config: EnvConfig,
    episodes: int = 25,
    max_steps: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    set_global_seed(seed)
    env = make_env(env_config)
    rows: list[dict[str, float]] = []

    try:
        for episode in range(1, episodes + 1):
            metrics = agent.evaluate_episode(env, max_steps=max_steps)
            metrics["evaluation_episode"] = episode
            rows.append(metrics)
    finally:
        env.close()

    return pd.DataFrame(rows)


def train_agent(
    agent,
    env_config: EnvConfig,
    training_config: TrainingConfig,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_global_seed(seed)
    env = make_env(env_config)
    history_rows: list[dict[str, float]] = []
    evaluation_rows: list[dict[str, float]] = []

    try:
        for episode in range(1, training_config.episodes + 1):
            metrics = agent.train_episode(env, max_steps=training_config.max_steps)
            metrics["episode"] = episode
            history_rows.append(metrics)

            if training_config.eval_every and episode % training_config.eval_every == 0:
                evaluation = evaluate_agent(
                    agent,
                    env_config=env_config,
                    episodes=training_config.eval_episodes,
                    max_steps=training_config.max_steps,
                    seed=seed + episode,
                )
                evaluation_rows.append(
                    {
                        "episode": episode,
                        "mean_eval_return": float(evaluation["episode_return"].mean()),
                        "std_eval_return": float(evaluation["episode_return"].std(ddof=0)),
                        "mean_eval_score": float(evaluation["score"].mean()),
                        "std_eval_score": float(evaluation["score"].std(ddof=0)),
                    }
                )
    finally:
        env.close()

    return pd.DataFrame(history_rows), pd.DataFrame(evaluation_rows)


def run_parameter_sweep(
    agent_factory: Callable[..., object],
    sweep_values: Iterable[dict[str, float]],
    env_config: EnvConfig,
    training_config: TrainingConfig,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    summary_rows: list[dict[str, float | str]] = []
    details: dict[str, dict[str, pd.DataFrame]] = {}

    for sweep_index, params in enumerate(sweep_values):
        label = ", ".join(f"{key}={value}" for key, value in params.items())
        agent = agent_factory(**params)
        history, evaluation = train_agent(
            agent,
            env_config=env_config,
            training_config=training_config,
            seed=seed + 1000 * sweep_index,
        )

        if evaluation.empty:
            evaluation = evaluate_agent(
                agent,
                env_config=env_config,
                episodes=training_config.eval_episodes,
                max_steps=training_config.max_steps,
                seed=seed + 1000 * sweep_index,
            )
            final_mean_return = float(evaluation["episode_return"].mean())
            final_mean_score = float(evaluation["score"].mean())
        else:
            final_mean_return = float(evaluation.iloc[-1]["mean_eval_return"])
            final_mean_score = float(evaluation.iloc[-1]["mean_eval_score"])

        summary_rows.append(
            {
                "label": label,
                "mean_eval_return": final_mean_return,
                "mean_eval_score": final_mean_score,
            }
        )
        details[label] = {"history": history, "evaluation": evaluation}

    return pd.DataFrame(summary_rows), details


def transfer_evaluation(
    agent,
    env_configs: Iterable[EnvConfig],
    episodes: int = 25,
    max_steps: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for index, env_config in enumerate(env_configs):
        evaluation = evaluate_agent(
            agent,
            env_config=env_config,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed + index,
        )
        rows.append(
            {
                "environment": env_config.label,
                "height": env_config.height,
                "width": env_config.width,
                "pipe_gap": env_config.pipe_gap,
                "mean_return": float(evaluation["episode_return"].mean()),
                "mean_score": float(evaluation["score"].mean()),
            }
        )

    return pd.DataFrame(rows)
