#!/usr/bin/env python3
"""
Prototype visual clustering demo to create per-identity contact sheets.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from unlabeled_media_tagger.pipeline.embed_faces import embed_faces_in_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster faces and generate contact sheets from a directory of images."
    )
    parser.add_argument("--input-dir", required=True, help="Root directory of images")
    parser.add_argument(
        "--out-dir",
        default="outputs/clusters",
        help="Output directory for clusters and montages",
    )
    parser.add_argument(
        "--detector-backend",
        default="retinaface",
        help="DeepFace detector backend",
    )
    parser.add_argument(
        "--model-name",
        default="ArcFace",
        help="DeepFace embedding model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.67,
        help="Cosine similarity threshold for clustering",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.995,
        help="Minimum face detection confidence",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=40,
        help="Minimum min(w,h) of clipped face bbox",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=160,
        help="Tile size for montage grid",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=6,
        help="Number of columns in montage grid",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit on number of images to process",
    )
    parser.add_argument(
        "--gray-threshold",
        type=float,
        default=20.0,
        help="Skip face crops that look grayscale (lower = stricter, higher = more skipping)",
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.20,
        help="Cosine similarity threshold for merging clusters after greedy pass",
    )
    parser.add_argument(
        "--min-merge-count",
        type=int,
        default=1,
        help="Only merge clusters if each has at least this many faces",
    )
    parser.add_argument(
        "--csv-out",
        default="outputs/clusters/cluster_results.csv",
        help="Output CSV file path for cluster results",
    )
    parser.add_argument(
        "--include-embeddings",
        action="store_true",
        help="If set, store embedding vectors in CSV as JSON strings",
    )
    return parser.parse_args()


def find_images(root_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in root_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]
    )


def colorfulness_score(bgr_crop: np.ndarray) -> float:
    b, g, r = cv2.split(bgr_crop.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)

    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)

    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def clip_bbox_to_image(
    bbox: Dict[str, int], width: int, height: int
) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(0, int(bbox["x"]))
    y1 = max(0, int(bbox["y"]))
    x2 = min(width, int(bbox["x"]) + int(bbox["w"]))
    y2 = min(height, int(bbox["y"]) + int(bbox["h"]))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def is_grayscale_like(bgr_crop: np.ndarray, threshold: float = 10.0) -> bool:
    # Hasler–Süsstrunk-ish colorfulness
    b, g, r = cv2.split(bgr_crop.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)

    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)

    colorfulness = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    return colorfulness < threshold


def save_face_crop(
    image_bgr: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    output_path: Path,
) -> None:
    x1, y1, x2, y2 = crop_box
    crop_bgr = image_bgr[y1:y2, x1:x2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_img = Image.fromarray(crop_rgb)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop_img.save(output_path, format="JPEG", quality=92)


def merge_clusters_in_place(
    clusters: list[dict],
    merge_threshold: float,
    min_merge_count: int = 2,
) -> list[dict]:
    """
    Repeatedly merge the most similar pair of clusters while similarity >= merge_threshold.
    Assumes each cluster has: sum (np.ndarray), mean (np.ndarray normalized), count (int),
    faces (list), source_images (set).
    Returns a NEW list of clusters (ids will be reassigned later if you want).
    """
    # Work on a copy of list (dicts still mutable, which is fine here)
    clusters = list(clusters)

    changed = True
    while changed:
        changed = False
        best_i, best_j = None, None
        best_sim = merge_threshold

        # Find best merge candidate
        for i in range(len(clusters)):
            ci = clusters[i]
            if ci["count"] < min_merge_count:
                continue
            for j in range(i + 1, len(clusters)):
                cj = clusters[j]
                if cj["count"] < min_merge_count:
                    continue
                if ci["source_images"].intersection(cj["source_images"]):
                    continue
                sim = cosine_similarity(ci["mean"], cj["mean"])
                if sim >= best_sim:
                    best_sim = sim
                    best_i, best_j = i, j

        # Merge if we found a pair
        if best_i is not None and best_j is not None:
            a = clusters[best_i]
            b = clusters[best_j]

            a["sum"] += b["sum"]
            a["count"] += b["count"]
            mean = a["sum"] / a["count"]
            norm = np.linalg.norm(mean)
            a["mean"] = mean / (norm + 1e-12)

            a["faces"].extend(b["faces"])
            a["source_images"].update(b["source_images"])

            # remove b (delete higher index first)
            del clusters[best_j]
            changed = True

    return clusters


def write_csv_results(
    clusters: List[Dict],
    csv_out_path: Path,
    include_embeddings: bool = False,
) -> None:
    """Write cluster results to CSV file."""
    csv_out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "cluster_id",
        "source_file",
        "crop_path",
        "confidence",
        "assigned_sim",
        "final_sim",
    ]
    if include_embeddings:
        fieldnames.append("embedding")
    
    with open(csv_out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for cluster in clusters:
            cluster_id = cluster.get("id", "unknown")
            for face in cluster["faces"]:
                row = {
                    "cluster_id": cluster_id,
                    "source_file": face.get("source_relpath", face.get("source_stem", "")),
                    "crop_path": face.get("crop_path", ""),
                    "confidence": face.get("confidence", ""),
                    "assigned_sim": face.get("assigned_sim", ""),
                    "final_sim": face.get("final_sim", ""),
                }
                if include_embeddings:
                    emb = face.get("embedding", [])
                    if isinstance(emb, np.ndarray):
                        row["embedding"] = json.dumps(emb.tolist())
                    else:
                        row["embedding"] = json.dumps(emb) if emb else ""
                writer.writerow(row)


def create_montage(
    faces: List[Dict],
    montage_path: Path,
    tile_size: int,
    cols: int,
) -> None:
    if not faces:
        return

    rows = int(math.ceil(len(faces) / cols))
    montage = Image.new("RGB", (cols * tile_size, rows * tile_size), color=(0, 0, 0))
    draw = ImageDraw.Draw(montage)

    for idx, face in enumerate(faces):
        row = idx // cols
        col = idx % cols
        x = col * tile_size
        y = row * tile_size

        crop_img = Image.open(face["crop_path"]).convert("RGB")
        tile = ImageOps.pad(crop_img, (tile_size, tile_size), color=(0, 0, 0))
        montage.paste(tile, (x, y))

        label = face.get("label")
        if label:
            text_y = y + tile_size - 16
            draw.rectangle(
                [(x, text_y), (x + tile_size, text_y + 16)],
                fill=(0, 0, 0),
            )
            draw.text((x + 4, text_y + 2), label, fill=(255, 255, 255))

    montage_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(montage_path, format="JPEG", quality=92)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    image_paths = find_images(input_dir)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    clusters: List[Dict] = []
    total_faces = 0
    processed_images = 0

    for image_path in image_paths:
        faces = embed_faces_in_image(
            str(image_path),
            detector_backend=args.detector_backend,
            model_name=args.model_name,
        )

        if not faces:
            processed_images += 1
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            processed_images += 1
            continue

        height, width = image_bgr.shape[:2]
        # Calculate source_relpath from input_dir
        try:
            source_relpath = str(image_path.relative_to(input_dir))
        except ValueError:
            source_relpath = str(image_path)
        
        for face in faces:
            confidence = float(face.get("confidence", 0.0))
            bbox_clipped = face.get("bbox_clipped", {})
            if confidence < args.min_confidence:
                continue

            face_w = int(bbox_clipped.get("w", 0))
            face_h = int(bbox_clipped.get("h", 0))
            if min(face_w, face_h) < args.min_face_size:
                continue

            crop_box = clip_bbox_to_image(bbox_clipped, width, height)
            if crop_box is None:
                continue

            # Crop pixels for grayscale check (and reuse later if you want)
            x1, y1, x2, y2 = crop_box
            face_crop = image_bgr[y1:y2, x1:x2]
            cf = colorfulness_score(face_crop)
            if face_crop.size == 0:
                continue

            if is_grayscale_like(face_crop, threshold=args.gray_threshold):
                continue

            embedding = np.array(face.get("embedding", []), dtype=np.float32)
            if embedding.size == 0:
                continue
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            # embedding = np.array(face.get("embedding", []), dtype=np.float32)
            # if embedding.size == 0:
            #     continue

            best_cluster = None
            best_sim = -1.0
            for cluster in clusters:
                if str(image_path) in cluster["source_images"]:
                    continue
                sim = cosine_similarity(embedding, cluster["mean"])
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = cluster

            if best_cluster is None or best_sim < args.threshold:
                cluster_id = len(clusters) + 1
                best_cluster = {
                    "id": cluster_id,
                    "sum": embedding.copy(),
                    "mean": embedding.copy(),
                    "count": 1,
                    "faces": [],
                    "source_images": {str(image_path)},
                }
                clusters.append(best_cluster)
                best_sim = 1.0
            else:
                best_cluster["sum"] += embedding
                best_cluster["count"] += 1
                mean = best_cluster["sum"] / best_cluster["count"]
                norm = np.linalg.norm(mean)
                best_cluster["mean"] = mean / (norm + 1e-12) # if norm > 0 else mean
                # best_cluster["mean"] = best_cluster["sum"] / best_cluster["count"]

            total_faces += 1
            crop_name = f"{total_faces:05d}_{image_path.stem}.jpg"
            cluster_dir = out_dir / f"cluster_{best_cluster['id']:03d}"
            crop_path = cluster_dir / crop_name
            save_face_crop(image_bgr, crop_box, crop_path)

            face_record = {
                "crop_path": str(crop_path),
                "source_stem": image_path.stem,
                "source_relpath": source_relpath,
                "confidence": confidence,
                "cf": cf,
                "embedding": embedding,
                "assigned_sim": best_sim,
            }
            best_cluster["faces"].append(face_record)
            best_cluster["source_images"].add(str(image_path))

        processed_images += 1

    # --- NEW: merge clusters after greedy assignment ---
    clusters = merge_clusters_in_place(
        clusters,
        merge_threshold=args.merge_threshold,
        min_merge_count=args.min_merge_count,
    )

    # Optional: reassign ids to be clean/contiguous after merges
    for idx, cluster in enumerate(clusters, start=1):
        cluster["id"] = idx

    for cluster in clusters:
        for face in cluster["faces"]:
            final_sim = cosine_similarity(face["embedding"], cluster["mean"])
            face["final_sim"] = final_sim
            face["label"] = f"sim={final_sim:.2f}"

    for cluster in clusters:
        montage_path = out_dir / f"cluster_{cluster['id']:03d}_montage.jpg"
        create_montage(
            cluster["faces"],
            montage_path,
            tile_size=args.tile_size,
            cols=args.cols,
        )

    # Write CSV results
    csv_out_path = Path(args.csv_out)
    write_csv_results(clusters, csv_out_path, include_embeddings=args.include_embeddings)
    print(f"\nCSV export written to: {csv_out_path}")

    print("Clustering summary")
    print(f"  Total images: {processed_images}")
    print(f"  Total faces: {total_faces}")
    print(f"  Num clusters: {len(clusters)}")
    if clusters:
        counts = ", ".join(str(len(c["faces"])) for c in clusters)
        print(f"  Faces per cluster: {counts}")


if __name__ == "__main__":
    main()
