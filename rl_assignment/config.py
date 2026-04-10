from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    height: int = 15
    width: int = 20
    pipe_gap: int = 4

    def as_kwargs(self) -> dict[str, int]:
        return {
            "height": self.height,
            "width": self.width,
            "pipe_gap": self.pipe_gap,
        }

    @property
    def label(self) -> str:
        return (
            f"height={self.height}, width={self.width}, pipe_gap={self.pipe_gap}"
        )


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 4000
    max_steps: int = 500
    eval_every: int = 200
    eval_episodes: int = 25
    moving_average_window: int = 100
