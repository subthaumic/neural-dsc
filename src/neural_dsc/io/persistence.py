"""
Data persistence utilities for experiment scripts.

This module provides helper functions for saving and loading experiment data,
enabling visualization regeneration without rerunning expensive DSC construction.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..utils.jaccard_index import (
    JACCARD_FILTRATION_INTERVALS,
    JACCARD_FILTRATION_POINTS,
    JACCARD_METHOD_ID,
)


def save_pvalues(pvalues: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save p-value data to JSON file.

    Args:
        pvalues: List of p-value dictionaries from DSC construction
        filepath: Path to save JSON file
    """
    # Convert numpy arrays to lists for JSON serialization
    pvalues_serializable = []
    for pv in pvalues:
        pv_dict = {
            'predictor': list(pv['predictor']),
            'target': int(pv['target']),
            'dimension': int(pv['dimension']),
            'min_pvalue': float(pv['min_pvalue']),
            'all_pvalues': pv['all_pvalues'].tolist() if isinstance(pv['all_pvalues'], np.ndarray) else list(pv['all_pvalues']),
        }
        pvalues_serializable.append(pv_dict)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(pvalues_serializable, f, indent=2)
    print(f"  Saved p-values to {filepath}")


def load_pvalues(filepath: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load p-value data from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        List of p-value dictionaries (with numpy arrays restored), or None if file doesn't exist
    """
    if not Path(filepath).exists():
        return None

    with open(filepath, 'r') as f:
        pvalues_serializable = json.load(f)

    # Convert lists back to numpy arrays where appropriate
    pvalues = []
    for pv_dict in pvalues_serializable:
        pv = {
            'predictor': tuple(pv_dict['predictor']),
            'target': pv_dict['target'],
            'dimension': pv_dict['dimension'],
            'min_pvalue': pv_dict['min_pvalue'],
            'all_pvalues': np.array(pv_dict['all_pvalues']),
        }
        pvalues.append(pv)

    print(f"  Loaded {len(pvalues)} p-value sets from {filepath}")
    return pvalues


def save_probe_data(probe_data_list: List[torch.Tensor], filepath: str) -> None:
    """
    Save list of probe data tensors.

    Args:
        probe_data_list: List of torch tensors (one per class)
        filepath: Path to save PT file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe_data_list, filepath)
    print(f"  Saved probe data to {filepath}")


def load_probe_data(filepath: str) -> Optional[List[torch.Tensor]]:
    """
    Load probe data tensors.

    Args:
        filepath: Path to PT file

    Returns:
        List of torch tensors, or None if file doesn't exist
    """
    if not Path(filepath).exists():
        return None

    probe_data = torch.load(filepath, weights_only=False)
    print(f"  Loaded probe data from {filepath}")
    return probe_data


def save_jaccard_data(jaccard_matrix: np.ndarray, jaccard_by_dim: Dict[int, Any], filepath: str) -> None:
    """
    Save Jaccard analysis to JSON.

    Args:
        jaccard_matrix: 2D numpy array of Jaccard distances
        jaccard_by_dim: Dictionary mapping dimension -> value
        filepath: Path to save JSON file
    """
    data = {
        'method': {
            'id': JACCARD_METHOD_ID,
            'quantity': 'distance',
            'filtration_domain': [0.0, 1.0],
            'filtration_intervals': JACCARD_FILTRATION_INTERVALS,
            'filtration_points': JACCARD_FILTRATION_POINTS,
            'vertices_included': False,
        },
        'matrix': jaccard_matrix.tolist(),
        'by_dimension': {str(k): float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in jaccard_by_dim.items()}
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved Jaccard analysis to {filepath}")


def load_jaccard_data(filepath: str) -> Tuple[Optional[np.ndarray], Optional[Dict[int, Any]]]:
    """
    Load Jaccard analysis from JSON.

    Args:
        filepath: Path to JSON file

    Returns:
        Tuple of (jaccard_matrix, jaccard_by_dim), or (None, None) if file doesn't exist
    """
    if not Path(filepath).exists():
        return None, None

    with open(filepath, 'r') as f:
        data = json.load(f)

    method = data.get('method', {})
    if (
        method.get('id') != JACCARD_METHOD_ID
        or method.get('quantity') != 'distance'
        or method.get('filtration_points') != JACCARD_FILTRATION_POINTS
        or method.get('vertices_included') is not False
    ):
        print(f"  Ignoring legacy Jaccard cache at {filepath}")
        return None, None

    jaccard_matrix = np.array(data['matrix'])
    jaccard_by_dim = {int(k): v for k, v in data['by_dimension'].items()}

    print(f"  Loaded Jaccard analysis from {filepath}")
    return jaccard_matrix, jaccard_by_dim


def save_sweep_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save hyperparameter sweep results to pickle file.

    Args:
        results: Dictionary of sweep results
        filepath: Path to save pickle file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"  Saved sweep results to {filepath}")


def load_sweep_results(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load hyperparameter sweep results from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Dictionary of sweep results, or None if file doesn't exist
    """
    if not Path(filepath).exists():
        return None

    with open(filepath, 'rb') as f:
        results = pickle.load(f)

    print(f"  Loaded sweep results from {filepath}")
    return results


def save_metadata(metadata_dict: Dict[str, Any], filepath: str) -> None:
    """
    Save experiment metadata to JSON.

    Args:
        metadata_dict: Dictionary of metadata (accuracy, classes, etc.)
        filepath: Path to save JSON file
    """
    # Convert numpy types to Python types
    serializable_dict = {}
    for key, value in metadata_dict.items():
        if isinstance(value, (np.integer, np.floating)):
            serializable_dict[key] = value.item()
        elif isinstance(value, np.ndarray):
            serializable_dict[key] = value.tolist()
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (np.integer, np.floating)):
            serializable_dict[key] = [v.item() if isinstance(v, (np.integer, np.floating)) else v for v in value]
        else:
            serializable_dict[key] = value

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(serializable_dict, f, indent=2)
    print(f"  Saved metadata to {filepath}")


def load_metadata(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load experiment metadata from JSON.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary of metadata, or None if file doesn't exist
    """
    if not Path(filepath).exists():
        return None

    with open(filepath, 'r') as f:
        metadata = json.load(f)

    print(f"  Loaded metadata from {filepath}")
    return metadata


def check_config_fingerprint(config: Dict[str, Any], output_dir, strict: bool = True) -> bool:
    """
    Refuse to reuse cached outputs that were produced under a different config.

    These scripts are resumable: they load `.pkl` sweep results and `.txt` DSCs
    whenever the files exist. That is only safe while the configuration is
    unchanged. It silently was not once before -- a sweep rerun at
    `max_dimension = 7` replayed the entire monosemantic half from `.pkl`
    checkpoints written at `max_dimension = 4`, and published the result under
    metadata claiming 7. Nothing in the output recorded the mismatch.

    This writes the configuration alongside the outputs on the first run and
    compares against it on every later one, so a changed constant fails loudly
    instead of producing a half-stale result set.

    Args:
        config: the run configuration; must be JSON-serialisable.
        output_dir: directory holding the cached outputs.
        strict: raise on mismatch (default). False only reports.

    Returns:
        True if the fingerprint matched or was newly written, False on mismatch
        when `strict` is False.

    Raises:
        RuntimeError: on mismatch when `strict`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config_fingerprint.json"

    payload = json.loads(json.dumps(config, sort_keys=True, default=str))
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if not path.exists():
        with open(path, "w") as f:
            json.dump({"sha256": digest, "config": payload}, f, indent=2, sort_keys=True)
        return True

    with open(path) as f:
        stored = json.load(f)
    if stored.get("sha256") == digest:
        return True

    old = stored.get("config", {})
    changed = sorted(
        set(old) | set(payload),
        key=lambda k: str(k),
    )
    lines = [
        f"    {k}: {old.get(k, '<absent>')!r} -> {payload.get(k, '<absent>')!r}"
        for k in changed
        if old.get(k) != payload.get(k)
    ]
    message = (
        f"Cached outputs in {output_dir} were produced under a different "
        f"configuration:\n" + "\n".join(lines) + "\n"
        f"Reusing them would mix incompatible results under the new metadata. "
        f"Delete the stale outputs (including any .pkl and .txt checkpoints) "
        f"and rerun, or restore the previous configuration."
    )
    if strict:
        raise RuntimeError(message)
    print(f"WARNING: {message}")
    return False
