from dataclasses import dataclass, asdict
import argparse, json
from typing import List, Union
import torch

@dataclass
class Args:
    dataset: Union[None, str] = None
    train_dataset_path: Union[None, str] = None
    valid_dataset_path: Union[None, str] = None
    model_path: Union[None, str] = None
    save_path: Union[None, str] = None

    pai: int = 0.99

    input_dim: int = 1
    hidden1_dim: int = 8
    hidden2_dim: int = 8
    use_feature: bool = True

    num_headers: int = 8
    ep: float = 1e-4
    num_epoch: int = 100
    learning_rate: float = 1e-3

    supervised_rate: float = 0.9

    sample_num: int = 100
    batch_size: int = 1

    log_interval: int = 10
    step_checkpointing: bool = False
    checkpointing_steps: int = 100

    step_evaluation: bool = False
    evaluation_steps: int = 100

    epoch_checkpointing: bool = False
    epoch_evaluation: bool = False

    early_stopping: bool = False
    early_stopping_stall_num: int = 5

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str, default=None)
    parsed = parser.parse_args()
    with open(parsed.c, 'r') as f:
        c = json.load(f)
    args = Args(**c)

    return args