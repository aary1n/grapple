"""Prototypical network calibration for the reflexive path.

From vla-architecture.md §5:
    - Compute and store frozen anchor embeddings from a pre-trained embedding network
    - Classify at runtime via cosine distance to prototypes
    - NO weight updates — the embedding model is frozen
    - 5-10 anchor gestures are sufficient (centroids in embedding space)
    - Lookup adds <0.5ms to the reflexive budget
    - Storage: prototypes_{user_id}_v{N}.npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PrototypeSet:
    """A set of gesture prototypes for a single user."""

    # (num_classes, embed_dim) — one centroid per gesture class
    centroids: np.ndarray
    # Class labels corresponding to each centroid row
    class_labels: list[str]
    # Metadata
    user_id: str
    version: int
    num_shots_per_class: list[int]  # how many anchors contributed to each centroid


class PrototypicalCalibrator:
    """Compute and store prototype embeddings for personalized gesture classification.

    Usage:
        1. User performs anchor gestures during calibration
        2. Each gesture is passed through the frozen embedding model
        3. Embeddings are averaged per-class to form prototypes (centroids)
        4. At runtime, new gestures are classified by nearest centroid (cosine distance)

    The embedding model is NEVER modified. This is a forward-pass-only operation.
    """

    def __init__(self, embed_dim: int = 128):
        self.embed_dim = embed_dim

    def compute_prototypes(
        self,
        embeddings: dict[str, torch.Tensor],
        user_id: str,
        version: int = 1,
    ) -> PrototypeSet:
        """Compute prototype centroids from labeled anchor embeddings.

        Args:
            embeddings: {class_label: (N_shots, embed_dim)} — embeddings per class.
                        These should already be augmented per architecture rules.
            user_id: User identifier for storage.
            version: Prototype version number.

        Returns:
            PrototypeSet with computed centroids.
        """
        centroids = []
        class_labels = []
        num_shots = []

        for label, emb in embeddings.items():
            if emb.ndim == 1:
                emb = emb.unsqueeze(0)

            # L2-normalize before averaging (unit hypersphere)
            emb_norm = F.normalize(emb, p=2, dim=-1)
            centroid = emb_norm.mean(dim=0)
            # Re-normalize the centroid
            centroid = F.normalize(centroid, p=2, dim=-1)

            centroids.append(centroid.cpu().numpy())
            class_labels.append(label)
            num_shots.append(emb.shape[0])

        return PrototypeSet(
            centroids=np.stack(centroids, axis=0),
            class_labels=class_labels,
            user_id=user_id,
            version=version,
            num_shots_per_class=num_shots,
        )

    def classify(
        self,
        embedding: torch.Tensor,
        prototypes: PrototypeSet,
    ) -> tuple[str, float]:
        """Classify a gesture by cosine similarity to prototypes.

        Args:
            embedding: (embed_dim,) or (1, embed_dim) — single gesture embedding
            prototypes: Precomputed prototype set

        Returns:
            (predicted_class_label, confidence_score)
        """
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)

        # Normalize query
        query = F.normalize(embedding, p=2, dim=-1)  # (1, embed_dim)

        # Cosine similarity to all centroids
        centroids_t = torch.from_numpy(prototypes.centroids).to(
            device=query.device, dtype=query.dtype
        )  # (num_classes, embed_dim)
        similarities = (query @ centroids_t.T).squeeze(0)  # (num_classes,)

        # Best match
        best_idx = similarities.argmax().item()
        confidence = similarities[best_idx].item()

        return prototypes.class_labels[best_idx], confidence

    @staticmethod
    def save(prototypes: PrototypeSet, path: str | Path) -> None:
        """Save prototypes to .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            centroids=prototypes.centroids,
            class_labels=np.array(prototypes.class_labels),
            user_id=prototypes.user_id,
            version=prototypes.version,
            num_shots_per_class=np.array(prototypes.num_shots_per_class),
        )

    @staticmethod
    def load(path: str | Path) -> PrototypeSet:
        """Load prototypes from .npz file."""
        data = np.load(path, allow_pickle=False)
        return PrototypeSet(
            centroids=data["centroids"],
            class_labels=data["class_labels"].tolist(),
            user_id=str(data["user_id"]),
            version=int(data["version"]),
            num_shots_per_class=data["num_shots_per_class"].tolist(),
        )
