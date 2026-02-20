#!/usr/bin/env python3
"""
Face Embedding Similarity Demo

This script demonstrates face embedding extraction and similarity matching
across a folder of images.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from unlabeled_media_tagger.pipeline.embed_faces import embed_faces_in_image


def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Cosine similarity score (0 to 1, higher is more similar)
    """
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def find_images(input_dir: Path) -> List[Path]:
    """
    Find all image files in the directory.
    
    Args:
        input_dir: Directory to search
    
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
    
    return sorted(image_files)


def main():
    parser = argparse.ArgumentParser(
        description='Extract face embeddings and find similar faces across images'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing images to process'
    )
    parser.add_argument(
        '--detector-backend',
        type=str,
        default='retinaface',
        help='Face detector backend (default: retinaface)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='ArcFace',
        help='Embedding model name (default: ArcFace)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top matches to display (default: 10)'
    )
    parser.add_argument(
        '--query',
        type=int,
        default=0,
        help='Index of query face in scan order (default: 0, first detected face)'
    )
    parser.add_argument(
        '--exclude-same-image',
        action='store_true',
        default=True,
        help='Exclude matches from the same source image as query (default: True)'
    )
    parser.add_argument(
        '--include-same-image',
        action='store_false',
        dest='exclude_same_image',
        help='Include matches from the same source image as query'
    )
    parser.add_argument(
        '--exclude-self',
        action='store_true',
        default=True,
        help='Exclude the exact query face from results (default: True)'
    )
    parser.add_argument(
        '--include-self',
        action='store_false',
        dest='exclude_self',
        help='Include the exact query face in results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(name)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format='%(name)s - %(levelname)s - %(message)s'
        )
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Directory not found: {args.input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: Not a directory: {args.input_dir}")
        sys.exit(1)
    
    print(f"Face Embedding Similarity Demo")
    print(f"=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Detector: {args.detector_backend}")
    print(f"Embedding model: {args.model_name}")
    print(f"Top-K: {args.top_k}")
    print(f"Query index: {args.query}")
    print(f"Exclude same image: {args.exclude_same_image}")
    print(f"Exclude self: {args.exclude_self}")
    print()
    
    # Find all images
    print("Finding images...")
    image_files = find_images(input_dir)
    
    if len(image_files) == 0:
        print("Error: No image files found in directory")
        sys.exit(1)
    
    print(f"Found {len(image_files)} image(s)")
    print()
    
    # Extract embeddings from all images
    print("Extracting face embeddings...")
    all_faces = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"  [{i}/{len(image_files)}] Processing {image_path.name}...", end='')
        
        try:
            faces = embed_faces_in_image(
                str(image_path),
                detector_backend=args.detector_backend,
                model_name=args.model_name
            )
            
            all_faces.extend(faces)
            print(f" {len(faces)} face(s)")
            
        except Exception as e:
            print(f" Error: {e}")
    
    print()
    print(f"Total faces detected: {len(all_faces)}")
    
    # Count unique images
    unique_images = set(face['source_path'] for face in all_faces)
    print(f"Unique images with faces: {len(unique_images)}")
    
    if len(unique_images) == 1:
        print("\nNote: All faces are from a single image. Only one image provided; Cross-image matching isn't possible. Same-image matching is enabled/disabled based on your flags.")
        print("For meaningful cross-image matches, provide a directory with multiple images.")
    
    if len(all_faces) == 0:
        print("No faces found in any images. Exiting.")
        sys.exit(0)
    
    # Validate and select query face
    if args.query < 0 or args.query >= len(all_faces):
        print(f"Error: Query index {args.query} out of range (0-{len(all_faces)-1})")
        sys.exit(1)
    
    query_face = all_faces[args.query]
    query_embedding = query_face['embedding']
    query_source = query_face['source_path']
    
    print()
    print("Query face:")
    print(f"  Index: {args.query}")
    print(f"  Source: {Path(query_face['source_path']).name}")
    print(f"  BBox: x={query_face['bbox']['x']}, y={query_face['bbox']['y']}, "
          f"w={query_face['bbox']['w']}, h={query_face['bbox']['h']}")
    print(f"  Confidence: {query_face['confidence']:.4f}")
    print()
    
    # Compute similarities to all faces
    print("Computing similarities...")
    similarities = []
    
    for i, face in enumerate(all_faces):
        # Apply filtering
        if args.exclude_self and i == args.query:
            continue
        if args.exclude_same_image and face['source_path'] == query_source:
            continue
        
        similarity = cosine_similarity(query_embedding, face['embedding'])
        similarities.append({
            'index': i,
            'face': face,
            'similarity': similarity
        })
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Display top-k matches
    top_k = min(args.top_k, len(similarities))
    
    print()
    if len(similarities) == 0:
        print("No matching faces found with current filter settings.")
    else:
        print(f"Top {top_k} most similar faces:")
        print("=" * 60)
        
        for rank, match in enumerate(similarities[:top_k], 1):
            face = match['face']
            similarity = match['similarity']
            
            print(f"\nRank {rank}:")
            print(f"  Similarity: {similarity:.4f}")
            print(f"  Index: {match['index']}")
            print(f"  Image: {Path(face['source_path']).name}")
            print(f"  BBox: x={face['bbox']['x']}, y={face['bbox']['y']}, "
                  f"w={face['bbox']['w']}, h={face['bbox']['h']}")
            print(f"  Confidence: {face['confidence']:.4f}")


if __name__ == '__main__':
    main()
