"""
ACDC (Automatic Circuit DisCovery) for MLP models

This module implements the ACDC algorithm from Conmy et al. (2023):
"Towards Automated Circuit Discovery for Mechanistic Interpretability"
https://arxiv.org/abs/2304.14997

The algorithm uses activation patching between clean and corrupted data to identify
important computational circuits. Specifically, it:

1. Starts with clean data (good performance)
2. For each edge, tests the RELATIVE damage caused by corrupting it on top of current circuit H
3. Removes edges that cause minimal relative damage (below threshold)
4. Uses additive patching to preserve other connections to each neuron
5. Iterates in reverse-topological order for efficiency

Key implementation details following Conmy et al. (2023):
- Direction: Corrupted-into-clean patching (not clean-into-corrupted)
- Method: Additive updates (not overwriting entire neurons)
- Scoring: Relative damage against current circuit H (not absolute damage)
- Reference: All previously removed edges stay corrupted during testing
- Order: Reverse-topological traversal (not random sampling)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import copy

from .graph import hidden_edges


class ACDCResult:
    """Container for ACDC algorithm results."""
    
    def __init__(self):
        self.edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()  # ((layer_from, neuron_from), (layer_to, neuron_to)) pairs
        self.edge_scores: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = {}
        self.final_metric: float = 0.0
        self.iteration_history: List[Dict] = []
    
    def __repr__(self):
        return f"ACDCResult(edges={len(self.edges)}, final_metric={self.final_metric:.4f})"


class ACDC:
    """
    ACDC implementation for MLP models using activation patching.
    
    Discovers important connections by measuring the damage caused when 
    individual edges are corrupted in an otherwise clean forward pass.
    
    This implementation follows the algorithm from Conmy et al. (2023):
    - Uses corrupted-into-clean patching direction
    - Applies additive updates to preserve other connections
    - Measures damage (not improvement) for edge scoring
    - Iterates in reverse-topological order for efficiency
    
    Note: Both input and output layers are treated as "external" and are not 
    included in the discovered circuits. This makes ACDC results more comparable 
    with DSC (Directed Simplicial Complex) analysis which focuses on internal 
    computational structure between hidden layers.
    """
    
    def __init__(self, model, threshold: float = 0.01, metric: str = "kl_div"):
        """
        Initialize ACDC.
        
        Args:
            model: MLP model to analyze
            threshold: Threshold below which edges are removed
            metric: Metric to use ("kl_div", "mse", "accuracy")
        """
        self.model = model
        self.threshold = threshold
        self.metric = metric
        self.device = next(model.parameters()).device
        
        # Track which edges are currently included - now neuron-to-neuron
        self.active_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        self._initialize_edges()
        
        # Cache for activations - we need both clean and corrupted
        self.clean_cache = {}
        self.corrupted_cache = {}
        
    def _initialize_edges(self):
        """Initialize all neuron-to-neuron edges in the MLP, excluding edges from input and to output layers."""
        self.active_edges = hidden_edges(self.model)
    
    def _get_layer_activations(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Post-activation output of each hidden layer, plus the logits.

        Keyed by hidden-layer index, so `activations[i]` is what
        `MLP.forward` calls `activations[i]`, and `activations[n_hidden]` is
        the output layer. These values are what an edge out of a neuron
        actually multiplies, so they have to be read *after* the
        nonlinearity: an `nn.Linear` forward hook returns the pre-activation
        instead, and patching with pre-activations silently reconstructs a
        different network -- on the handcrafted monosemantic net, patching
        every edge then lands on [0, 0, 9.75, 3.0] where the corrupted run
        gives [0, 0, 3.0, 0.75].
        """
        with torch.no_grad():
            output, hidden = self.model(x, return_activations=True)

        activations = {i: h.detach().clone() for i, h in hidden.items()}
        activations[len(self.model.hidden_layers)] = output.detach().clone()
        return activations
    
    def _compute_metric(self, clean_logits: torch.Tensor, patched_logits: torch.Tensor) -> float:
        """Compute comparison metric between clean and patched outputs."""
        if self.metric == "kl_div":
            clean_probs = F.softmax(clean_logits, dim=-1)
            patched_log_probs = F.log_softmax(patched_logits, dim=-1)
            # batchmean, not mean: 'mean' also divides by the number of classes,
            # so the reported divergence -- and hence the meaning of
            # `self.threshold` -- would depend on the output width.
            return F.kl_div(patched_log_probs, clean_probs, reduction='batchmean').item()
        elif self.metric == "mse":
            return F.mse_loss(clean_logits, patched_logits).item()
        elif self.metric == "accuracy":
            clean_pred = torch.argmax(clean_logits, dim=-1)
            patched_pred = torch.argmax(patched_logits, dim=-1)
            return (clean_pred != patched_pred).float().mean().item()
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _run_with_activation_patching(self, 
                                    clean_data: torch.Tensor,
                                    corrupted_data: torch.Tensor,
                                    clean_activations: Dict[int, torch.Tensor],
                                    corrupted_activations: Dict[int, torch.Tensor],
                                    edges_to_corrupt: Set[Tuple[Tuple[int, int], Tuple[int, int]]]) -> torch.Tensor:
        """
        Run model on clean data while patching specific edges with corrupted activations.
        
        This implements ACDC: Start with clean data (good performance) and for each 
        edge being tested, replace its contribution with the corrupted activation to measure
        the damage caused by removing that edge.
        
        Edge format: ((from_layer, from_neuron), (to_layer, to_neuron))
        Where layer 0 = input, layer 1 = first hidden, etc.
        """
        
        # Start with clean input data
        current = clean_data

        with torch.no_grad():
            # Process each hidden layer
            for layer_idx, layer in enumerate(self.model.hidden_layers):
                # Apply the layer transformation normally (clean path)
                # This computes: layer(current) = W @ current + b
                next_activation = layer(current)

                # NOW: For specific edges targeting this layer, patch with corrupted values
                for edge in edges_to_corrupt:
                    (from_layer, from_neuron), (to_layer, to_neuron) = edge

                    # Check if this edge targets the current layer we're computing
                    if to_layer == layer_idx + 1:  # layer_idx+1 because layer 0 is input
                        # Get the weight for this specific connection
                        weight = layer.weight[to_neuron, from_neuron]

                        # The source of an edge into layer_idx+1 is a neuron of
                        # layer_idx, whose activation under the *current circuit*
                        # is `current` -- the value this pass has already
                        # computed, with every upstream patch folded in. Reading
                        # the clean side from the unpatched cache instead would
                        # make the patch non-recursive: patching an edge would
                        # not propagate through the edges downstream of it, and
                        # the circuit would not be a circuit.
                        clean_source = current[:, from_neuron]
                        if from_layer == 0:
                            corrupted_source = corrupted_data[:, from_neuron]
                        else:
                            source_layer_idx = from_layer - 1
                            if source_layer_idx not in corrupted_activations:
                                # Fallback: skip this edge if we don't have the activations
                                continue
                            corrupted_source = corrupted_activations[source_layer_idx][:, from_neuron]

                        # Apply additive patch: replace clean contribution with corrupted contribution
                        # This preserves all other incoming connections to this neuron
                        delta = weight * (corrupted_source - clean_source)
                        next_activation[:, to_neuron] += delta

                # Apply activation function
                current = torch.relu(next_activation)

            # Final output layer
            output = self.model.output_layer(current)

        return output
    
    def discover_circuit(self, 
                        clean_data: torch.Tensor, 
                        corrupted_data: torch.Tensor,
                        max_iterations: int = 100) -> ACDCResult:
        """
        Run ACDC algorithm to discover important circuit using activation patching.
        
        This implements the ACDC algorithm from Conmy et al. (2023):
        1. Start with clean data (good performance)
        2. For each edge, test RELATIVE damage caused by corrupting it on top of current circuit H
        3. Remove edges that cause relative damage below threshold (least important)
        4. Iterate in reverse-topological order for efficiency
        
        Key: When testing edge e, we patch ALL previously removed edges PLUS e,
        then measure damage relative to current circuit H (not full clean model).
        
        Args:
            clean_data: Input data where model performs well
            corrupted_data: Input data where model performs poorly
            max_iterations: Maximum number of iterations
            
        Returns:
            ACDCResult containing discovered edges and metrics
        """
        result = ACDCResult()
        current_edges = self.active_edges.copy()
        
        # Get baseline activations and outputs
        self.model.eval()
        with torch.no_grad():
            clean_output = self.model(clean_data)
            corrupted_output = self.model(corrupted_data)
        
        # Cache both clean and corrupted activations for patching
        clean_activations = self._get_layer_activations(clean_data)
        corrupted_activations = self._get_layer_activations(corrupted_data)
        
        # Group edges by target layer for reverse-topological order
        edges_by_target_layer = defaultdict(list)
        for edge in current_edges:
            (from_layer, from_neuron), (to_layer, to_neuron) = edge
            edges_by_target_layer[to_layer].append(edge)
        
        iteration = 0
        # Process layers in reverse order (output to input)
        layer_order = sorted(edges_by_target_layer.keys(), reverse=True)
        
        for target_layer in layer_order:
            if iteration >= max_iterations:
                break
                
            layer_edges = [e for e in edges_by_target_layer[target_layer] if e in current_edges]
            
            while layer_edges and iteration < max_iterations:
                edge_damages = {}
                
                # Get baseline performance of current circuit H (with all previously removed edges corrupted)
                removed_edges = self.active_edges - current_edges
                if removed_edges:
                    current_circuit_output = self._run_with_activation_patching(
                        clean_data, corrupted_data, clean_activations, 
                        corrupted_activations, removed_edges
                    )
                    baseline_damage = self._compute_metric(clean_output, current_circuit_output)
                else:
                    # No edges removed yet, baseline is clean performance
                    baseline_damage = 0.0
                
                # Test each edge in this layer by measuring damage when corrupted
                for edge in layer_edges:
                    # Test what happens if we corrupt this edge ON TOP OF all previously removed edges
                    edges_to_corrupt = removed_edges | {edge}
                    
                    # Run clean data with this edge (and all previously removed edges) corrupted
                    patched_output = self._run_with_activation_patching(
                        clean_data, corrupted_data, clean_activations, 
                        corrupted_activations, edges_to_corrupt
                    )
                    
                    # Measure RELATIVE damage: how much additional damage this edge causes
                    # beyond what the current circuit H already experiences
                    absolute_damage = self._compute_metric(clean_output, patched_output)
                    relative_damage = absolute_damage - baseline_damage
                    edge_damages[edge] = relative_damage
                    # Keep the most recent damage for every edge, not only for
                    # the one removed. This is ACDC's filtration: the relative
                    # damage an edge was last measured to cause is the threshold
                    # at which it would have been removed, so the same run that
                    # yields a circuit at one threshold also yields a ranking of
                    # every edge -- which is what a DSC's filtration values can
                    # be compared against.
                    result.edge_scores[edge] = relative_damage
                
                # Find edge with smallest damage (least important)
                if not edge_damages:
                    break
                    
                min_edge = min(edge_damages.keys(), key=lambda k: edge_damages[k])
                min_damage = edge_damages[min_edge]
                
                # Record this iteration
                iteration_info = {
                    'iteration': iteration,
                    'target_layer': target_layer,
                    'edges_remaining': len(current_edges),
                    'min_edge': min_edge,
                    'min_damage': min_damage,
                    'baseline_damage': baseline_damage,
                    'tested_edges': len(edge_damages)
                }
                result.iteration_history.append(iteration_info)
                
                # If relative damage is below threshold, remove this edge
                if min_damage < self.threshold:
                    current_edges.remove(min_edge)
                    layer_edges.remove(min_edge)
                    print(f"Iteration {iteration}: Removed edge {min_edge} (relative damage: {min_damage:.6f}, baseline: {baseline_damage:.6f})")
                else:
                    print(f"Iteration {iteration}: No edges below threshold in layer {target_layer}. Moving to next layer.")
                    break
                
                iteration += 1
        
        # Store final results
        result.edges = current_edges
        
        # Compute final metric: how well does the discovered circuit perform?
        if current_edges:
            # Test the final circuit by corrupting all removed edges
            removed_edges = self.active_edges - current_edges
            final_patched_output = self._run_with_activation_patching(
                clean_data, corrupted_data, clean_activations, 
                corrupted_activations, removed_edges
            )
            result.final_metric = self._compute_metric(clean_output, final_patched_output)
        else:
            # If no edges remain, circuit performance equals corrupted performance
            result.final_metric = self._compute_metric(clean_output, corrupted_output)
        
        print(f"ACDC completed: {len(result.edges)} edges remaining, "
              f"final metric: {result.final_metric:.6f}")
        
        return result
    
    def create_corrupted_data(self, clean_data: torch.Tensor, corruption_type: str = "noise",
                              seed: Optional[int] = None) -> torch.Tensor:
        """
        Create corrupted version of clean data for ACDC analysis.

        Args:
            clean_data: Original clean input data
            corruption_type: Type of corruption ("noise", "shuffle", "zero", "flip")
            seed: RNG seed. Every corruption here is random, so without a seed
                the counterfactual whose effectiveness is measured is not the
                counterfactual the circuit is then discovered against, and no
                two runs of the script return the same circuit.

        Returns:
            Corrupted version of the input data
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=clean_data.device).manual_seed(int(seed))
        corrupted = clean_data.clone()
        
        if corruption_type == "noise":
            # Add Gaussian noise
            noise_std = 0.1 * torch.std(clean_data)
            noise = torch.randn(clean_data.shape, generator=generator,
                               dtype=clean_data.dtype, device=clean_data.device) * noise_std
            corrupted = clean_data + noise
            
        elif corruption_type == "shuffle":
            # Shuffle features within each sample
            for i in range(corrupted.shape[0]):
                perm = torch.randperm(corrupted.shape[1], generator=generator)
                corrupted[i] = corrupted[i][perm]
                
        elif corruption_type == "zero":
            # Zero out random features
            mask = torch.rand(corrupted.shape, generator=generator,
                             dtype=corrupted.dtype, device=corrupted.device) > 0.5
            corrupted = corrupted * mask.float()
            
        elif corruption_type == "flip":
            # Randomly flip signs of x, y, or both coordinates
            # This changes which quadrant points belong to, creating severe corruption
            # Following ACDC principle: ALL data points should be corrupted (no unchanged points)
            for i in range(corrupted.shape[0]):
                flip_choice = torch.randint(1, 4, (1,), generator=generator).item()  # 1: flip x, 2: flip y, 3: flip both (no option 0)
                
                if flip_choice == 1:  # Flip x coordinate (dimension 0)
                    corrupted[i, 0] = -corrupted[i, 0]
                elif flip_choice == 2:  # Flip y coordinate (dimension 1)
                    corrupted[i, 1] = -corrupted[i, 1]
                elif flip_choice == 3:  # Flip both coordinates
                    corrupted[i, 0] = -corrupted[i, 0]
                    corrupted[i, 1] = -corrupted[i, 1]
            
        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")
            
        return corrupted
    
    def get_circuit_summary(self, result: ACDCResult) -> Dict:
        """Get human-readable summary of discovered circuit."""
        summary = {
            'total_edges': len(self.active_edges),
            'remaining_edges': len(result.edges),
            'removed_edges': len(self.active_edges) - len(result.edges),
            'sparsity': 1.0 - len(result.edges) / len(self.active_edges),
            'final_metric': result.final_metric,
            'edge_count_by_layer': self._count_edges_by_layer(result.edges)
        }
        return summary
    
    def _count_edges_by_layer(self, edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]]) -> Dict:
        """Count remaining edges by layer connection."""
        counts = {}
        for edge in edges:
            (from_layer, _), (to_layer, _) = edge
            connection = f"L{from_layer}->L{to_layer}"
            counts[connection] = counts.get(connection, 0) + 1
        return counts
