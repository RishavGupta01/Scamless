from dataclasses import dataclass


@dataclass
class TrainConfig:
    backbone: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_len: int = 256
    epochs: int = 3
    batch_size: int = 32
    lr: float = 2e-5
    seed: int = 42
    adversarial_rate: float = 0.3  # fraction of labeled examples augmented per epoch
    output_dir: str = "artifacts/model_v1"
