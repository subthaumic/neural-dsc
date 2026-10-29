"""
Circuit-discovery baselines for MLP models.

`acdc` is the iterative activation-patching search of Conmy et al. (2023);
`eap` is the attribution-patching approximation of Syed et al. (2023) and its
integrated-gradients variant from Hanna et al. (2024). `graph` holds the edge
set both search and the translation into a DSC's flat vertex indices.
"""

from .acdc import ACDC, ACDCResult
from .eap import (
    edge_attribution_scores,
    kl_to_clean,
    target_logit_diff,
    top_k_circuit,
)
from .graph import (
    flat_index,
    hidden_edges,
    relation_kind,
    to_flat_edges,
)

__all__ = [
    "ACDC",
    "ACDCResult",
    "edge_attribution_scores",
    "kl_to_clean",
    "target_logit_diff",
    "top_k_circuit",
    "flat_index",
    "hidden_edges",
    "relation_kind",
    "to_flat_edges",
]
