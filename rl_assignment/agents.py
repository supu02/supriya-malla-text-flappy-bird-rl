from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict

import numpy as np

State = tuple[int, int]
EpisodeStep = tuple[State, int, float]


def as_state(observation: tuple[int, int]) -> State:
    return int(observation[0]), int(observation[1])


class BaseTabularAgent:
    def __init__(
        self,
        n_actions: int = 2,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.999,
        min_epsilon: float = 0.02,
        seed: int = 0,
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = np.random.default_rng(seed)
        self.q_values: DefaultDict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

    def action_values(self, state: State) -> np.ndarray:
        return self.q_values[state]

    def greedy_action(self, state: State) -> int:
        q_values = self.action_values(state)
        max_value = np.max(q_values)
        best_actions = np.flatnonzero(np.isclose(q_values, max_value))
        return int(self.rng.choice(best_actions))

    def select_action(self, state: State, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(state)

    def state_value(self, state: State) -> float:
        return float(np.max(self.action_values(state)))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def evaluate_episode(self, env, max_steps: int = 500) -> dict[str, float]:
        observation, info = env.reset()
        state = as_state(observation)
        total_reward = 0.0
        episode_length = 0
        terminated = False
        truncated = False

        for step in range(max_steps):
            action = self.select_action(state, explore=False)
            observation, reward, terminated, truncated, info = env.step(action)
            state = as_state(observation)
            total_reward += reward
            episode_length = step + 1
            if terminated or truncated:
                break

        return {
            "episode_return": float(total_reward),
            "episode_length": float(episode_length),
            "score": float(info["score"]),
            "terminated": float(terminated),
            "truncated": float(truncated),
        }


class MonteCarloControlAgent(BaseTabularAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.visit_counts: DefaultDict[tuple[State, int], int] = defaultdict(int)

    def train_episode(self, env, max_steps: int = 500) -> dict[str, float]:
        observation, info = env.reset()
        state = as_state(observation)
        episode: list[EpisodeStep] = []
        total_reward = 0.0
        episode_length = 0
        terminated = False
        truncated = False

        for step in range(max_steps):
            action = self.select_action(state, explore=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode.append((state, action, float(reward)))
            state = as_state(observation)
            total_reward += reward
            episode_length = step + 1
            if terminated or truncated:
                break

        self._update_from_episode(episode)
        self.decay_epsilon()

        return {
            "episode_return": float(total_reward),
            "episode_length": float(episode_length),
            "score": float(info["score"]),
            "epsilon": float(self.epsilon),
        }

    def _update_from_episode(self, episode: list[EpisodeStep]) -> None:
        if not episode:
            return

        returns = np.zeros(len(episode), dtype=np.float64)
        discounted_return = 0.0

        for index in range(len(episode) - 1, -1, -1):
            _, _, reward = episode[index]
            discounted_return = reward + self.gamma * discounted_return
            returns[index] = discounted_return

        visited_pairs: set[tuple[State, int]] = set()
        for (state, action, _), discounted_return in zip(episode, returns):
            key = (state, action)
            if key in visited_pairs:
                continue

            visited_pairs.add(key)
            self.visit_counts[key] += 1
            step_size = 1.0 / self.visit_counts[key]
            self.q_values[state][action] += (
                discounted_return - self.q_values[state][action]
            ) * step_size


class SarsaLambdaAgent(BaseTabularAgent):
    def __init__(
        self,
        alpha: float = 0.1,
        lambda_: float = 0.9,
        use_true_online: bool = False,
        trace_type: str = "replacing",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.lambda_ = lambda_
        self.use_true_online = use_true_online
        self.trace_type = trace_type

    def train_episode(self, env, max_steps: int = 500) -> dict[str, float]:
        if self.use_true_online:
            return self._train_true_online_episode(env, max_steps=max_steps)
        return self._train_classic_episode(env, max_steps=max_steps)

    def _train_classic_episode(self, env, max_steps: int = 500) -> dict[str, float]:
        observation, info = env.reset()
        state = as_state(observation)
        action = self.select_action(state, explore=True)
        traces: DefaultDict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

        total_reward = 0.0
        episode_length = 0

        for step in range(max_steps):
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = as_state(observation)
            done = bool(terminated or truncated)
            total_reward += reward
            episode_length = step + 1

            if done:
                td_target = reward
                next_action = None
            else:
                next_action = self.select_action(next_state, explore=True)
                td_target = reward + self.gamma * self.q_values[next_state][next_action]

            td_error = td_target - self.q_values[state][action]

            if self.trace_type == "accumulating":
                traces[state][action] += 1.0
            else:
                traces[state][action] = 1.0

            active_states = list(traces.keys())
            for traced_state in active_states:
                self.q_values[traced_state] += self.alpha * td_error * traces[traced_state]
                traces[traced_state] *= self.gamma * self.lambda_
                if np.all(np.abs(traces[traced_state]) < 1e-10):
                    del traces[traced_state]

            if done:
                break

            state = next_state
            action = int(next_action)

        self.decay_epsilon()

        return {
            "episode_return": float(total_reward),
            "episode_length": float(episode_length),
            "score": float(info["score"]),
            "epsilon": float(self.epsilon),
        }

    def _train_true_online_episode(self, env, max_steps: int = 500) -> dict[str, float]:
        observation, info = env.reset()
        state = as_state(observation)
        action = self.select_action(state, explore=True)
        traces: DefaultDict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=np.float64)
        )

        total_reward = 0.0
        episode_length = 0
        previous_q = 0.0

        for step in range(max_steps):
            current_q = self.q_values[state][action]
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = as_state(observation)
            done = bool(terminated or truncated)
            total_reward += reward
            episode_length = step + 1

            if done:
                next_q = 0.0
                next_action = None
            else:
                next_action = self.select_action(next_state, explore=True)
                next_q = self.q_values[next_state][next_action]

            td_error = reward + self.gamma * next_q - current_q

            active_trace = traces[state][action]
            for traced_state in list(traces.keys()):
                traces[traced_state] *= self.gamma * self.lambda_
                if np.all(np.abs(traces[traced_state]) < 1e-10):
                    del traces[traced_state]

            traces[state][action] += (
                1.0 - self.alpha * self.gamma * self.lambda_ * active_trace
            )

            correction = current_q - previous_q
            for traced_state in list(traces.keys()):
                self.q_values[traced_state] += (
                    self.alpha * (td_error + correction) * traces[traced_state]
                )

            self.q_values[state][action] -= self.alpha * correction
            previous_q = next_q

            if done:
                break

            state = next_state
            action = int(next_action)

        self.decay_epsilon()

        return {
            "episode_return": float(total_reward),
            "episode_length": float(episode_length),
            "score": float(info["score"]),
            "epsilon": float(self.epsilon),
        }
