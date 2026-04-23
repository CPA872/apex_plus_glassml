"""Pluggable expert routing distributions for MoE AllToAll demand matrices.

Each distribution class implements a single interface: given a source GPU index
and the number of tokens it holds, return the number of tokens routed to each
destination GPU in the group.  The caller (_fill_alltoall) uses these counts
to compute byte traffic.

To add a new routing model:
  1. Subclass ExpertDistribution
  2. Implement route_tokens(src_idx, num_tokens, group_size) -> np.ndarray
  3. Register it in DISTRIBUTION_REGISTRY
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class ExpertDistribution(ABC):
    """Base class for expert routing distributions."""

    @abstractmethod
    def route_tokens(
        self, src_idx: int, num_tokens: int, group_size: int
    ) -> np.ndarray:
        """Return token counts routed from src GPU to each of `group_size` destinations.

        Args:
            src_idx: Index of the source GPU within its group (0-based).
            num_tokens: Number of tokens on this source GPU.
            group_size: Number of GPUs in the AllToAll group.

        Returns:
            1-D array of length group_size where counts[j] = tokens sent to GPU j.
            Must sum to num_tokens.
        """
        ...


class UniformDistribution(ExpertDistribution):
    """Uniform routing: each destination GPU receives ~equal tokens.

    Uses multinomial sampling so each source GPU gets slightly different
    counts (realistic noise), but expected value is uniform.
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)

    def route_tokens(
        self, src_idx: int, num_tokens: int, group_size: int
    ) -> np.ndarray:
        probs = np.ones(group_size) / group_size
        return self._rng.multinomial(num_tokens, probs)


class ZipfDistribution(ExpertDistribution):
    """Zipfian expert popularity: popular experts attract more tokens.

    Experts are assigned to GPUs round-robin by popularity rank (same as
    the simulator's _zipf_skew convention). The Zipf exponent `s` controls
    skew: 0 = uniform, 0.3-0.5 = typical with load-balancing loss, 1.0 =
    classic Zipf (heavy skew).

    All source GPUs share the same destination probabilities (the popularity
    distribution is a global property of the model), but multinomial sampling
    gives each source GPU different realized counts.
    """

    def __init__(self, num_experts: int, zipf_s: float = 1.0, seed: int = 42):
        self._num_experts = num_experts
        self._zipf_s = zipf_s
        self._rng = np.random.default_rng(seed)
        self._prob_cache: dict[int, np.ndarray] = {}

    def _gpu_probs(self, group_size: int) -> np.ndarray:
        """Per-GPU routing probability under Zipfian expert popularity."""
        if group_size in self._prob_cache:
            return self._prob_cache[group_size]

        if self._zipf_s <= 0.0 or group_size <= 1:
            probs = np.ones(group_size) / group_size
        else:
            ranks = np.arange(1, self._num_experts + 1, dtype=np.float64)
            raw = 1.0 / np.power(ranks, self._zipf_s)
            gpu_loads = np.zeros(group_size)
            for g in range(group_size):
                gpu_loads[g] = raw[g::group_size].sum()
            probs = gpu_loads / gpu_loads.sum()

        self._prob_cache[group_size] = probs
        return probs

    def route_tokens(
        self, src_idx: int, num_tokens: int, group_size: int
    ) -> np.ndarray:
        probs = self._gpu_probs(group_size)
        return self._rng.multinomial(num_tokens, probs)


class DirichletDistribution(ExpertDistribution):
    """Each source GPU draws its own routing probability vector from Dirichlet(alpha).

    Unlike Zipf (where all sources share the same global popularity vector),
    Dirichlet gives each source GPU an *independent* probability vector sampled
    from Dirichlet([alpha, ..., alpha]).  This models learned routers with
    heterogeneous per-GPU gating preferences.

    Parameter guide:
        alpha < 1  : spiky routing (tokens concentrate on a few GPUs)
        alpha = 1  : uniform over the simplex (each draw is random but varied)
        alpha > 10 : nearly uniform routing (low variance between GPUs)
    """

    def __init__(self, alpha: float = 1.0, seed: int = 42):
        self._alpha = alpha
        self._rng = np.random.default_rng(seed)

    def route_tokens(
        self, src_idx: int, num_tokens: int, group_size: int
    ) -> np.ndarray:
        probs = self._rng.dirichlet([self._alpha] * group_size)
        return self._rng.multinomial(num_tokens, probs)


# ---------------------------------------------------------------------------
# Registry: maps CLI name -> (constructor, description)
# ---------------------------------------------------------------------------

DISTRIBUTION_REGISTRY: dict[str, tuple[type, str]] = {
    "uniform": (UniformDistribution, "Equal probability per destination GPU"),
    "zipf": (ZipfDistribution, "Zipf-ranked expert popularity (param = exponent s)"),
    "dirichlet": (DirichletDistribution, "Per-GPU Dirichlet-sampled routing (param = alpha)"),
}


def make_expert_dist(
    name: str,
    param: float,
    num_experts: Optional[int] = None,
    seed: int = 42,
) -> ExpertDistribution:
    """Factory: create an ExpertDistribution from a CLI name + parameter.

    Args:
        name: Distribution name (key in DISTRIBUTION_REGISTRY).
        param: Distribution-specific parameter:
            - uniform: ignored
            - zipf: Zipf exponent s (0 = uniform, 1.0 = classic Zipf)
            - dirichlet: concentration alpha (< 1 spiky, 1 uniform simplex, > 10 flat)
        num_experts: Number of MoE experts (required for zipf, ignored otherwise).
        seed: RNG seed for reproducibility.
    """
    if name not in DISTRIBUTION_REGISTRY:
        available = ", ".join(sorted(DISTRIBUTION_REGISTRY))
        raise ValueError(f"Unknown distribution '{name}'. Available: {available}")

    if name == "uniform":
        return UniformDistribution(seed=seed)
    elif name == "zipf":
        if num_experts is None:
            raise ValueError("zipf distribution requires num_experts (from model config)")
        return ZipfDistribution(num_experts, zipf_s=param, seed=seed)
    elif name == "dirichlet":
        return DirichletDistribution(alpha=param, seed=seed)

    raise ValueError(f"No factory logic for distribution '{name}'")
