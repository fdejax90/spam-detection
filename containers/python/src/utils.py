import random

import numpy as np

# import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

SCORING = {
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
    "accuracy": accuracy_score,
}


def set_seed(seed) -> None:
    """Fix random seeds"""
    random.seed(seed)
    np.random.seed(seed)
