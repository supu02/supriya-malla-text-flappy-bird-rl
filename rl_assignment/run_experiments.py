from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
MPL_CONFIG_DIR = ROOT / "rl_assignment" / ".mplconfig"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_assignment import (
    EnvConfig,
    MonteCarloControlAgent,
    SarsaLambdaAgent,
    TrainingConfig,
    run_parameter_sweep,
    train_agent,
    transfer_evaluation,
)
from rl_assignment.plotting import (
    plot_parameter_sweep,
    plot_training_history,
    plot_value_heatmaps,
)


def _save_figure(path: Path) -> None:
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "results"
    output_dir.mkdir(exist_ok=True)

    env_config = EnvConfig(height=15, width=20, pipe_gap=4)
    baseline_training = TrainingConfig(
        episodes=2000,
        max_steps=200,
        eval_every=200,
        eval_episodes=30,
    )
    sweep_training = TrainingConfig(
        episodes=800,
        max_steps=200,
        eval_every=100,
        eval_episodes=20,
    )

    mc_agent = MonteCarloControlAgent(
        gamma=0.99,
        epsilon=0.20,
        epsilon_decay=0.999,
        min_epsilon=0.01,
        seed=1,
    )
    sarsa_agent = SarsaLambdaAgent(
        alpha=0.15,
        lambda_=0.9,
        gamma=0.99,
        epsilon=0.10,
        epsilon_decay=0.999,
        min_epsilon=0.02,
        seed=2,
    )

    mc_history, mc_eval = train_agent(
        mc_agent,
        env_config=env_config,
        training_config=baseline_training,
        seed=42,
    )
    sarsa_history, sarsa_eval = train_agent(
        sarsa_agent,
        env_config=env_config,
        training_config=baseline_training,
        seed=84,
    )

    mc_history.to_csv(output_dir / "mc_history.csv", index=False)
    mc_eval.to_csv(output_dir / "mc_eval.csv", index=False)
    sarsa_history.to_csv(output_dir / "sarsa_history.csv", index=False)
    sarsa_eval.to_csv(output_dir / "sarsa_eval.csv", index=False)

    plot_training_history(
        mc_history,
        mc_eval,
        window=baseline_training.moving_average_window,
        title="Monte Carlo Control on TextFlappyBird-v0",
    )
    _save_figure(output_dir / "mc_training.png")

    plot_training_history(
        sarsa_history,
        sarsa_eval,
        window=baseline_training.moving_average_window,
        title="Sarsa(lambda) on TextFlappyBird-v0",
    )
    _save_figure(output_dir / "sarsa_training.png")

    plot_value_heatmaps(mc_agent, env_config)
    _save_figure(output_dir / "mc_value_heatmaps.png")

    plot_value_heatmaps(sarsa_agent, env_config)
    _save_figure(output_dir / "sarsa_value_heatmaps.png")

    mc_sweep_values = [
        {
            "gamma": 0.99,
            "epsilon": 0.05,
            "epsilon_decay": 0.999,
            "min_epsilon": 0.02,
            "seed": 1,
        },
        {
            "gamma": 0.99,
            "epsilon": 0.10,
            "epsilon_decay": 0.999,
            "min_epsilon": 0.02,
            "seed": 1,
        },
        {
            "gamma": 0.99,
            "epsilon": 0.20,
            "epsilon_decay": 0.999,
            "min_epsilon": 0.02,
            "seed": 1,
        },
    ]
    sarsa_sweep_values = [
        {
            "alpha": 0.05,
            "lambda_": 0.7,
            "gamma": 0.99,
            "epsilon": 0.10,
            "epsilon_decay": 0.999,
            "min_epsilon": 0.02,
            "seed": 2,
        },
        {
            "alpha": 0.10,
            "lambda_": 0.8,
            "gamma": 0.99,
            "epsilon": 0.10,
            "epsilon_decay": 0.999,
            "min_epsilon": 0.02,
            "seed": 2,
        },
        {
            "alpha": 0.15,
            "lambda_": 0.9,
            "gamma": 0.99,
            "epsilon": 0.10,
            "epsilon_decay": 0.999,
            "min_epsilon": 0.02,
            "seed": 2,
        },
    ]

    mc_sweep, _ = run_parameter_sweep(
        MonteCarloControlAgent,
        mc_sweep_values,
        env_config=env_config,
        training_config=sweep_training,
        seed=10,
    )
    sarsa_sweep, _ = run_parameter_sweep(
        SarsaLambdaAgent,
        sarsa_sweep_values,
        env_config=env_config,
        training_config=sweep_training,
        seed=20,
    )
    mc_sweep.to_csv(output_dir / "mc_sweep.csv", index=False)
    sarsa_sweep.to_csv(output_dir / "sarsa_sweep.csv", index=False)

    plot_parameter_sweep(mc_sweep, title="Monte Carlo Parameter Sweep")
    _save_figure(output_dir / "mc_sweep.png")

    plot_parameter_sweep(sarsa_sweep, title="Sarsa(lambda) Parameter Sweep")
    _save_figure(output_dir / "sarsa_sweep.png")

    transfer_configs = [
        EnvConfig(height=15, width=20, pipe_gap=4),
        EnvConfig(height=18, width=20, pipe_gap=4),
        EnvConfig(height=15, width=24, pipe_gap=4),
        EnvConfig(height=15, width=20, pipe_gap=5),
    ]
    mc_transfer = transfer_evaluation(
        mc_agent,
        env_configs=transfer_configs,
        episodes=30,
        max_steps=200,
        seed=101,
    )
    sarsa_transfer = transfer_evaluation(
        sarsa_agent,
        env_configs=transfer_configs,
        episodes=30,
        max_steps=200,
        seed=202,
    )
    mc_transfer.to_csv(output_dir / "mc_transfer.csv", index=False)
    sarsa_transfer.to_csv(output_dir / "sarsa_transfer.csv", index=False)

    summary_lines = [
        "Baseline environment: " + env_config.label,
        "",
        "Monte Carlo final evaluation:",
        mc_eval.tail(1).to_string(index=False),
        "",
        "Sarsa(lambda) final evaluation:",
        sarsa_eval.tail(1).to_string(index=False),
        "",
        "Monte Carlo transfer:",
        mc_transfer.to_string(index=False),
        "",
        "Sarsa(lambda) transfer:",
        sarsa_transfer.to_string(index=False),
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines))

    print(f"Saved experiment outputs to {output_dir}")


if __name__ == "__main__":
    main()
