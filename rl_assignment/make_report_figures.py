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
import pandas as pd

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl_assignment import EnvConfig, MonteCarloControlAgent, SarsaLambdaAgent, TrainingConfig, train_agent
from rl_assignment.plotting import plot_report_q_and_sweeps, plot_training_comparison


def save_current(path: Path) -> None:
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parent
    results_dir = root / "results"

    mc_history = pd.read_csv(results_dir / "mc_history.csv")
    mc_eval = pd.read_csv(results_dir / "mc_eval.csv")
    mc_sweep = pd.read_csv(results_dir / "mc_sweep.csv")
    sarsa_history = pd.read_csv(results_dir / "sarsa_history.csv")
    sarsa_eval = pd.read_csv(results_dir / "sarsa_eval.csv")
    sarsa_sweep = pd.read_csv(results_dir / "sarsa_sweep.csv")

    plot_training_comparison(
        mc_history,
        mc_eval,
        sarsa_history,
        sarsa_eval,
        window=100,
    )
    save_current(results_dir / "report_training_compare.png")

    env_config = EnvConfig(height=15, width=20, pipe_gap=4)
    training_config = TrainingConfig(
        episodes=2000,
        max_steps=200,
        eval_every=0,
        eval_episodes=0,
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

    train_agent(mc_agent, env_config=env_config, training_config=training_config, seed=42)
    train_agent(sarsa_agent, env_config=env_config, training_config=training_config, seed=84)

    plot_report_q_and_sweeps(
        mc_agent,
        sarsa_agent,
        env_config,
        mc_sweep,
        sarsa_sweep,
    )
    save_current(results_dir / "report_q_and_sweeps.png")

    print("Saved compact report figures.")


if __name__ == "__main__":
    main()
