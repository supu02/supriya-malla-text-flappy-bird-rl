from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rl_assignment.config import EnvConfig
from rl_assignment.experiments import make_env


def moving_average(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, min_periods=1).mean()


def plot_training_history(
    history: pd.DataFrame,
    evaluation: pd.DataFrame | None = None,
    window: int = 100,
    title: str | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["episode"], history["episode_return"], alpha=0.25, label="Episode return")
    axes[0].plot(
        history["episode"],
        moving_average(history["episode_return"], window=window),
        linewidth=2,
        label=f"Moving average ({window})",
    )
    if evaluation is not None and not evaluation.empty:
        axes[0].plot(
            evaluation["episode"],
            evaluation["mean_eval_return"],
            marker="o",
            linewidth=1.5,
            label="Greedy evaluation",
        )
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].legend()

    axes[1].plot(history["episode"], history["score"], alpha=0.25, label="Episode score")
    axes[1].plot(
        history["episode"],
        moving_average(history["score"], window=window),
        linewidth=2,
        label=f"Moving average ({window})",
    )
    if evaluation is not None and not evaluation.empty:
        axes[1].plot(
            evaluation["episode"],
            evaluation["mean_eval_score"],
            marker="o",
            linewidth=1.5,
            label="Greedy evaluation",
        )
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def _state_axes(env_config: EnvConfig) -> tuple[np.ndarray, np.ndarray]:
    env = make_env(env_config)
    try:
        x_space, y_space = env.observation_space.spaces
        x_values = np.arange(x_space.start, x_space.start + x_space.n)
        y_values = np.arange(y_space.start, y_space.start + y_space.n)
    finally:
        env.close()
    return x_values, y_values


def _state_value_frame(agent, env_config: EnvConfig, action: int | None) -> pd.DataFrame:
    x_values, y_values = _state_axes(env_config)
    frame = pd.DataFrame(index=y_values, columns=x_values, dtype=float)

    for y_value in y_values:
        for x_value in x_values:
            state = (int(x_value), int(y_value))
            if state not in agent.q_values:
                continue
            if action is None:
                frame.loc[y_value, x_value] = agent.state_value(state)
            else:
                frame.loc[y_value, x_value] = agent.action_values(state)[action]

    return frame.sort_index(ascending=False)


def plot_value_heatmaps(agent, env_config: EnvConfig, cmap: str = "viridis"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    frames = [
        ("Q(s, idle)", _state_value_frame(agent, env_config, action=0)),
        ("Q(s, flap)", _state_value_frame(agent, env_config, action=1)),
        ("V(s)", _state_value_frame(agent, env_config, action=None)),
    ]

    for axis, (title, frame) in zip(axes, frames):
        sns.heatmap(frame, ax=axis, cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("dx")
        axis.set_ylabel("dy")

    return fig, axes


def plot_parameter_sweep(
    sweep_results: pd.DataFrame,
    metric: str = "mean_eval_return",
    title: str | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(sweep_results["label"], sweep_results[metric])
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Setting")
    ax.tick_params(axis="x", rotation=30)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_training_comparison(
    mc_history: pd.DataFrame,
    mc_eval: pd.DataFrame,
    sarsa_history: pd.DataFrame,
    sarsa_eval: pd.DataFrame,
    window: int = 100,
):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))

    axes[0].plot(
        mc_history["episode"],
        moving_average(mc_history["episode_return"], window=window),
        label="Monte Carlo",
        linewidth=2,
    )
    axes[0].plot(
        sarsa_history["episode"],
        moving_average(sarsa_history["episode_return"], window=window),
        label="Sarsa(lambda)",
        linewidth=2,
    )
    axes[0].scatter(
        mc_eval["episode"],
        mc_eval["mean_eval_return"],
        s=18,
        marker="o",
        alpha=0.8,
    )
    axes[0].scatter(
        sarsa_eval["episode"],
        sarsa_eval["mean_eval_return"],
        s=18,
        marker="s",
        alpha=0.8,
    )
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].set_title("Return", fontsize=10)
    axes[0].legend()

    axes[1].plot(
        mc_history["episode"],
        moving_average(mc_history["score"], window=window),
        label="Monte Carlo",
        linewidth=2,
    )
    axes[1].plot(
        sarsa_history["episode"],
        moving_average(sarsa_history["score"], window=window),
        label="Sarsa(lambda)",
        linewidth=2,
    )
    axes[1].scatter(
        mc_eval["episode"],
        mc_eval["mean_eval_score"],
        s=18,
        marker="o",
        alpha=0.8,
    )
    axes[1].scatter(
        sarsa_eval["episode"],
        sarsa_eval["mean_eval_score"],
        s=18,
        marker="s",
        alpha=0.8,
    )
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Score", fontsize=10)

    fig.tight_layout()
    return fig, axes


def plot_q_value_comparison(mc_agent, sarsa_agent, env_config: EnvConfig, cmap: str = "viridis"):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6), constrained_layout=True)
    frames = [
        ("Monte Carlo: Q(s, idle)", _state_value_frame(mc_agent, env_config, action=0)),
        ("Monte Carlo: Q(s, flap)", _state_value_frame(mc_agent, env_config, action=1)),
        ("Sarsa(lambda): Q(s, idle)", _state_value_frame(sarsa_agent, env_config, action=0)),
        ("Sarsa(lambda): Q(s, flap)", _state_value_frame(sarsa_agent, env_config, action=1)),
    ]

    for axis, (title, frame) in zip(axes.flat, frames):
        sns.heatmap(frame, ax=axis, cmap=cmap, cbar=False)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("dx")
        axis.set_ylabel("dy")

    return fig, axes


def plot_report_q_and_sweeps(
    mc_agent,
    sarsa_agent,
    env_config: EnvConfig,
    mc_sweep: pd.DataFrame,
    sarsa_sweep: pd.DataFrame,
    cmap: str = "viridis",
):
    fig, axes = plt.subplots(2, 2, figsize=(8.9, 5.7), constrained_layout=True)

    mc_q_flap = _state_value_frame(mc_agent, env_config, action=1)
    sarsa_q_flap = _state_value_frame(sarsa_agent, env_config, action=1)

    sns.heatmap(mc_q_flap, ax=axes[0, 0], cmap=cmap, cbar=False)
    axes[0, 0].set_title("Monte Carlo: Q(s, flap)", fontsize=9)
    axes[0, 0].set_xlabel("dx")
    axes[0, 0].set_ylabel("dy")

    sns.heatmap(sarsa_q_flap, ax=axes[0, 1], cmap=cmap, cbar=False)
    axes[0, 1].set_title("Sarsa(lambda): Q(s, flap)", fontsize=9)
    axes[0, 1].set_xlabel("dx")
    axes[0, 1].set_ylabel("dy")

    mc_labels = []
    for label in mc_sweep["label"]:
        if "epsilon=0.05" in label:
            mc_labels.append("0.05")
        elif "epsilon=0.1" in label:
            mc_labels.append("0.10")
        elif "epsilon=0.2" in label:
            mc_labels.append("0.20")
        else:
            mc_labels.append(label)
    axes[1, 0].bar(mc_labels, mc_sweep["mean_eval_score"])
    axes[1, 0].set_title("MC epsilon sweep", fontsize=9)
    axes[1, 0].set_xlabel("initial epsilon")
    axes[1, 0].set_ylabel("mean score")

    sarsa_labels = []
    for label in sarsa_sweep["label"]:
        if "alpha=0.05" in label and "lambda_=0.7" in label:
            sarsa_labels.append("a=.05\nl=.7")
        elif "alpha=0.1" in label and "lambda_=0.8" in label:
            sarsa_labels.append("a=.10\nl=.8")
        elif "alpha=0.15" in label and "lambda_=0.9" in label:
            sarsa_labels.append("a=.15\nl=.9")
        else:
            sarsa_labels.append(label)
    axes[1, 1].bar(sarsa_labels, sarsa_sweep["mean_eval_score"])
    axes[1, 1].set_title("Sarsa(lambda) sweep", fontsize=9)
    axes[1, 1].set_xlabel("alpha / lambda")
    axes[1, 1].set_ylabel("mean score")

    return fig, axes
