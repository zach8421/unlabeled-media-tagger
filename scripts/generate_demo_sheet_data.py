#!/usr/bin/env python3
"""
Demo Data Generator for spreadsheet team.

Runs the face detection and embedding pipeline on a directory of images
and outputs a UI-friendly CSV with face crops for clustering review.
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Import from the pipeline
from unlabeled_media_tagger.pipeline.embed_faces import embed_faces_in_image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate demo face data and CSV for spreadsheet team."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
        help="Directory of images to scan (required)"
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/demo_sheet",
        type=str,
        help="Output directory (default: outputs/demo_sheet)"
    )
    parser.add_argument(
        "--detector-backend",
        default="retinaface",
        type=str,
        help="Face detector backend (default: retinaface)"
    )
    parser.add_argument(
        "--model-name",
        default="ArcFace",
        type=str,
        help="Embedding model name (default: ArcFace)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.67,
        help="Cosine similarity threshold for clustering (default: 0.78)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.995,
        help="Minimum face detection confidence (default: 0.995)"
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=40,
        help="Minimum min(w,h) of clipped face bbox (default: 40)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit on number of images to process"
    )
    parser.add_argument(
        "--write-crops",
        action="store_true",
        default=True,
        help="Write face crops to out-dir/crops/ (default: True)"
    )
    parser.add_argument(
        "--csv-name",
        default="demo_faces.csv",
        type=str,
        help="Output CSV filename (default: demo_faces.csv)"
    )
    parser.add_argument(
        "--gray-threshold",
        type=float,
        default=20.0,
        help="Skip face crops that look grayscale (lower = stricter, higher = more skipping)",
    )
    return parser.parse_args()


def find_images(root_dir: Path) -> List[Path]:
    """Recursively find all image files in a directory."""
    return sorted([
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ])


def colorfulness_score(bgr_crop: np.ndarray) -> float:
    b, g, r = cv2.split(bgr_crop.astype(np.float32))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)

    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)

    return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def normalize_embedding(embedding: List[float]) -> np.ndarray:
    """Normalize embedding to unit length."""
    vec = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


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
    bbox_clipped: Dict[str, int],
    output_path: Path,
) -> None:
    """Save a face crop as JPEG."""
    x = int(bbox_clipped["x"])
    y = int(bbox_clipped["y"])
    w = int(bbox_clipped["w"])
    h = int(bbox_clipped["h"])
    
    x2 = x + w
    y2 = y + h
    
    crop_bgr = image_bgr[y:y2, x:x2]
    if crop_bgr.size == 0:
        logger.warning(f"Empty crop for {output_path}")
        return
    
    # Convert BGR to RGB and save
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    success = cv2.imwrite(str(output_path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
    if not success:
        logger.warning(f"Failed to save crop: {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Re-save with cv2.imwrite (already done above, but being explicit)


def main():
    """Main entry point."""
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1
    
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    stats = {
        "images_scanned": 0,
        "images_processed": 0,
        "faces_detected_raw": 0,
        "faces_kept": 0,
        "clusters_created": 0,
    }
    
    # Find all images
    image_paths = find_images(input_dir)
    total_images = len(image_paths)
    
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    
    logger.info(f"Found {total_images} images; processing {len(image_paths)}")
    
    # Clusters: list of dicts with:
    #   - mean: normalized embedding (np.ndarray)
    #   - faces: list of face records
    clusters = []
    
    # Process each image
    for idx, image_path in enumerate(image_paths, 1):
        logger.info(f"[{idx}/{len(image_paths)}] Processing {image_path.name}")
        
        try:
            # Load image
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                logger.warning(f"Failed to load image: {image_path}")
                continue
            
            stats["images_processed"] += 1
            
            # Detect and embed faces
            face_records = embed_faces_in_image(
                str(image_path),
                detector_backend=args.detector_backend,
                model_name=args.model_name
            )
            
            stats["faces_detected_raw"] += len(face_records)
            
            # Filter faces
            for face in face_records:
                # Check confidence
                if face["confidence"] < args.min_confidence:
                    continue
                
                # Check bbox_clipped exists and is valid
                if "bbox_clipped" not in face:
                    continue
                
                bbox_clipped = face["bbox_clipped"]
                min_dim = min(bbox_clipped["w"], bbox_clipped["h"])
                
                # Check min face size
                if min_dim < args.min_face_size:
                    continue
                
                # Crop pixels for grayscale check (and reuse later if you want)
                x = int(bbox_clipped["x"])
                y = int(bbox_clipped["y"])
                w = int(bbox_clipped["w"])
                h = int(bbox_clipped["h"])
                x2 = x + w
                y2 = y + h
                face_crop_bgr = image_bgr[y:y2, x:x2]
                
                # Check if crop looks grayscale
                if is_grayscale_like(face_crop_bgr, threshold=args.gray_threshold):
                    continue
                
                # Face passed filters
                stats["faces_kept"] += 1
                
                # Normalize embedding
                embedding_normalized = normalize_embedding(face["embedding"])
                
                # Greedy clustering: find best matching cluster
                best_cluster_idx = None
                best_similarity = args.threshold
                
                for cluster_idx, cluster in enumerate(clusters):
                    sim = cosine_similarity(embedding_normalized, cluster["mean"])
                    if sim >= best_similarity:
                        best_similarity = sim
                        best_cluster_idx = cluster_idx
                
                # Assign to cluster or create new one
                if best_cluster_idx is not None:
                    # Assign to existing cluster and update mean
                    cluster = clusters[best_cluster_idx]
                    cluster["faces"].append({
                        "embedding": embedding_normalized,
                        "source_path": image_path,
                        "bbox_clipped": bbox_clipped,
                        "confidence": face["confidence"],
                    })
                    # Update running mean
                    old_count = cluster["face_count"]
                    new_count = old_count + 1
                    cluster["mean"] = (
                        cluster["mean"] * old_count + embedding_normalized
                    ) / new_count
                    # Normalize the new mean
                    norm = np.linalg.norm(cluster["mean"])
                    if norm > 0:
                        cluster["mean"] = cluster["mean"] / norm
                    cluster["face_count"] = new_count
                else:
                    # Create new cluster
                    clusters.append({
                        "mean": embedding_normalized,
                        "faces": [{
                            "embedding": embedding_normalized,
                            "source_path": image_path,
                            "bbox_clipped": bbox_clipped,
                            "confidence": face["confidence"],
                        }],
                        "face_count": 1,
                    })
                    stats["clusters_created"] += 1
        
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}", exc_info=True)
            continue
    
    stats["images_scanned"] = total_images
    
    # Generate Face IDs and prepare CSV rows
    csv_rows = []
    face_id_counter = 1
    
    for cluster_idx, cluster in enumerate(clusters):
        for face in cluster["faces"]:
            face_id = f"F{face_id_counter:05d}"
            
            # Save crop if requested
            crop_relative_path = None
            if args.write_crops:
                source_stem = face["source_path"].stem
                crop_filename = f"{face_id}_{source_stem}.jpg"
                crop_path = crops_dir / crop_filename
                
                try:
                    image_bgr = cv2.imread(str(face["source_path"]))
                    if image_bgr is not None:
                        save_face_crop(image_bgr, face["bbox_clipped"], crop_path)
                        # Store relative path from out_dir
                        crop_relative_path = str(Path("crops") / crop_filename)
                except Exception as e:
                    logger.warning(f"Failed to save crop {crop_filename}: {e}")
            
            # Create CSV row
            row = {
                "Face ID": face_id,
                "Source File": face["source_path"].name,
                "Timestamp": "—",
                "Screenshot": crop_relative_path or "",
                "Suggested Cluster": str(cluster_idx + 1),
                "Confidence": f"{face['confidence']:.2f}",
                "Assigned Name": "",
            }
            csv_rows.append(row)
            face_id_counter += 1
    
    # Write CSV
    csv_path = out_dir / args.csv_name
    csv_fieldnames = [
        "Face ID",
        "Source File",
        "Timestamp",
        "Screenshot",
        "Suggested Cluster",
        "Confidence",
        "Assigned Name",
    ]
    
    try:
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info(f"CSV written to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to write CSV: {e}")
        return 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("DEMO DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total images scanned:     {stats['images_scanned']}")
    print(f"Images processed:         {stats['images_processed']}")
    print(f"Faces detected (raw):     {stats['faces_detected_raw']}")
    print(f"Faces kept (filtered):    {stats['faces_kept']}")
    print(f"Clusters created:         {stats['clusters_created']}")
    print(f"Rows written to CSV:      {len(csv_rows)}")
    print("=" * 60)
    print(f"Output directory:         {out_dir.resolve()}")
    print(f"CSV file:                 {csv_path.resolve()}")
    if args.write_crops:
        print(f"Face crops:               {crops_dir.resolve()}")
    print("=" * 60)
    
    # Print first 5 CSV rows
    if csv_rows:
        print("\nFirst 5 CSV rows:")
        print("-" * 60)
        for i, row in enumerate(csv_rows[:5]):
            print(f"Row {i+1}: {row}")
        print("-" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
