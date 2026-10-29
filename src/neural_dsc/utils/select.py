# Selection utilities for one-hot and two-hot logic
import torch
import numpy as np


def select_topk_by_cosine_similarity(outputs, X, target_vec, k):
    """
    Select top-k points whose output is closest (by cosine similarity) to the given target vector.
    target_vec should be a 1D tensor (e.g., one-hot or two-hot).
    """
    cos = torch.nn.functional.cosine_similarity(outputs, target_vec.unsqueeze(0), dim=1)
    topk_indices = torch.topk(cos, k=k).indices
    return X[topk_indices]


def get_twohot_indices(n_classes):
    """
    Return all unique pairs of class indices for two-hot selection.
    """
    indices = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            indices.append((i, j))
    return indices
