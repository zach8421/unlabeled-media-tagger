"""Compare Stage - face comparison and clustering."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt


class CompareStage:
    """
    Compare stage for face comparison and clustering.
    
    This stage is responsible for:
    - Comparing facial embeddings across different media files
    - Clustering similar faces together
    - Building a database of unique individuals
    - Tracking faces across media collection
    """
    
    def __init__(self, config=None):
        """
        Initialize the compare stage.
        
        Args:
            config: Configuration dictionary for comparison settings
        """
        self.config = config or {}
    
    def compare_faces(self, face_embeddings):
        """
        Compare face embeddings and cluster similar faces.
        
        Args:
            face_embeddings: List of facial embeddings to compare
            
        Returns:
            Dictionary mapping cluster IDs to lists of matching faces
            
        Raises:
            ValueError: If an embedding is malformed.
        """
        threshold = float(self.config.get("similarity_threshold", 0.68))
        clustered = defaultdict(list)
        centroids = []

        for face in face_embeddings:
            embedding = face.get("embedding")
            if not embedding:
                continue

            best_cluster = None
            best_similarity = -1.0
            for cluster_id, centroid in enumerate(centroids):
                similarity = cosine_similarity(embedding, centroid)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster_id

            if best_cluster is None or best_similarity < threshold:
                cluster_id = len(centroids)
                centroids.append([float(value) for value in embedding])
                assigned_similarity = 1.0
            else:
                cluster_id = best_cluster
                centroids[cluster_id] = update_centroid(
                    centroids[cluster_id],
                    embedding,
                    len(clustered[cluster_id]),
                )
                assigned_similarity = round(best_similarity, 6)

            clustered[cluster_id].append(
                {
                    **face,
                    "cluster_id": cluster_id,
                    "cluster_label": f"person_{cluster_id:03d}",
                    "similarity_to_cluster": assigned_similarity,
                }
            )

        return dict(clustered)
    
    def build_face_database(self, clustered_faces):
        """
        Build or update a database of unique individuals.
        
        Args:
            clustered_faces: Dictionary of clustered face data
            
        Returns:
            Updated face database
            
        Raises:
            None.
        """
        database = {}
        for cluster_id, faces in clustered_faces.items():
            database[cluster_id] = {
                "cluster_id": cluster_id,
                "cluster_label": f"person_{cluster_id:03d}",
                "face_count": len(faces),
                "drive_file_ids": sorted(
                    {face.get("drive_id") for face in faces if face.get("drive_id")}
                ),
                "media_names": sorted(
                    {face.get("media_name") for face in faces if face.get("media_name")}
                ),
            }

        return database


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two numeric vectors."""
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same length")

    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = sqrt(sum(float(a) * float(a) for a in left))
    right_norm = sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


def update_centroid(
    current_centroid: list[float],
    new_embedding: list[float],
    existing_count: int,
) -> list[float]:
    """Update a centroid with one new vector using an online mean."""
    next_count = existing_count + 1
    return [
        ((float(current) * existing_count) + float(new)) / next_count
        for current, new in zip(current_centroid, new_embedding)
    ]
