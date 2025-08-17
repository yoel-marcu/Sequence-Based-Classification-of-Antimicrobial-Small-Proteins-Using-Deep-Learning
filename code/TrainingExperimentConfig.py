from dataclasses import dataclass, field
from typing import Callable, Optional, Any, List, Type, Dict
import torch.nn as nn


@dataclass(order=True)
class TrainingExperimentConfig:
    """
    Configuration for a training experiment.
    Attributes:
        name (str): Name of the experiment.
        embedding_path (str): Path to the embedding dataset.
        model (nn.Module): The model to be trained.
        optimizer (Callable): Optimizer function for training.
        criterion (Callable): Loss function for training.
    """

    # Required (no defaults)
    name: str = field(compare=False)
    embedding_name: str = field(compare=False)
    embedding_path: str = field(compare=True)

    # Optional (with defaults)
    root: str = field(default=".", compare=False)
    pooling_fns: List[Callable] = field(default_factory=list, compare=False)
    network_type: str = field(default="fixed", compare=False)
    weighted: bool = field(default=False, compare=False)
    fixed_network: bool = field(default=False, compare=False)
    model_factory: Callable[[], nn.Module] = field(default=None, compare=False)
    model: Optional[nn.Module] = field(default=None, compare=False)
    optimizer: Optional[Callable] = field(default=None, compare=False)
    criterion: Optional[Callable] = field(default=None, compare=False)
    fold_preds: List = field(default_factory=list, compare=False)
    fold_labels: List = field(default_factory=list, compare=False)
    k_fold_predictions: Dict[str, Dict[str, Any]] = field(default_factory=dict, compare=False)
    final_results: Dict[str, Any] = field(default_factory=dict, compare=False)
