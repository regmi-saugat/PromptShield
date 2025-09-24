from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = field(
        default_factory=lambda: ["Wqkv", "Wo"]  # <-- ModernBERT
    )
    bias: str = "none"

@dataclass
class TrainConfig:
    model_name: str = "modernbert-base"
    max_samples: int = 1_000
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 1e-4
    warmup_steps: int = 50
    weight_decay: float = 0.01
    output_dir: str | None = None
    eval_strategy: str = "epoch"
    save_total_limit: int = 2
    metric_for_best: str = "f1"
    fp16: bool = True
    logging_steps: int = 10
