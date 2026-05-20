"""ColBERT / ColPali-style MaxSim on token-shaped embedding rows (pure numpy).

True ColPali aligns query transformer tokens vs document patch tokens produced by dedicated
Vision-Language encoders (multi-vector retrieval). Here we deterministically reshape a pooled
CLIP vector into pseudo-token rows for optional reranking (exploratory, not pretrained MaxSim).

References: Late interaction / MaxSim summaries in ColBERT, ColPali papers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """L2-normalise each matrix row."""

    import numpy as np_np

    norm = np_np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return mat / norm


def reshape_to_slots(vec: np.ndarray, slots: int) -> np.ndarray:
    """Reshape pooled ``(dim,)`` vector into ``(slots, dim // slots)``."""

    import numpy as np_np

    if vec.ndim != 1:
        msg = "expected 1D embedding"
        raise ValueError(msg)
    dim = vec.size
    if dim % slots != 0:
        msg = f"embedding dim {dim} not divisible into {slots} slots"
        raise ValueError(msg)
    mat = vec.reshape(slots, dim // slots)
    return mat.astype(np_np.float64, copy=False)


def maxsim_score(query_slots: np.ndarray, doc_slots: np.ndarray) -> float:
    """
    Approximate Σ_i max_j sim(q_i, d_j): rows are normalised pseudo-token embeddings.

    Shapes ``(tq, dim)``, ``(td, dim)``.
    """

    import numpy as np_np

    if query_slots.ndim != 2 or doc_slots.ndim != 2:
        msg = "query_slots and doc_slots must be rank-2 matrices"
        raise ValueError(msg)
    if query_slots.shape[1] != doc_slots.shape[1]:
        msg = "slot widths must align"
        raise ValueError(msg)
    qs = l2_normalize_rows(query_slots.astype(np_np.float64, copy=False))
    ds = l2_normalize_rows(doc_slots.astype(np_np.float64, copy=False))
    sims = qs @ ds.T  # shape (tq, td)
    return float(np_np.sum(np_np.max(sims, axis=1)))
