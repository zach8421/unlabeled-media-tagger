"""
Image Annotation Module

This module provides functionality to annotate images with face detection bounding boxes.
"""

import sys
from pathlib import Path
from typing import List, Dict
import cv2

from .detect_faces import detect_faces_in_image


def annotate_image(
    image_path: str,
    detections: List[Dict],
    output_path: str
) -> None:
    """
    Annotate an image with bounding boxes for detected faces.
    
    Args:
        image_path: Path to the input image file
        detections: List of face detections from detect_faces_in_image()
                   Each detection should have 'bbox' (x, y, w, h) and 'confidence'
        output_path: Path where the annotated image will be saved
    
    Raises:
        FileNotFoundError: If image_path does not exist
        ValueError: If image cannot be loaded
    """
    # Verify image exists
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Draw bounding boxes for each detection
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']
        
        # Extract coordinates
        x = int(bbox['x'])
        y = int(bbox['y'])
        w = int(bbox['w'])
        h = int(bbox['h'])
        
        # Draw rectangle (green color, 2px thickness)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw confidence score text
        confidence_text = f"{confidence:.2f}"
        # Position text above the bounding box
        text_y = y - 10 if y - 10 > 10 else y + h + 20
        
        # Add background rectangle for better text visibility
        text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            image, 
            (x, text_y - text_size[1] - 4), 
            (x + text_size[0], text_y + 4), 
            (0, 255, 0), 
            -1
        )
        
        # Draw text (black color on green background)
        cv2.putText(
            image,
            confidence_text,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save annotated image
    cv2.imwrite(str(output_path), image)
    print(f"Annotated image saved to: {output_path}")


def main():
    """CLI entry point for image annotation."""
    if len(sys.argv) < 2:
        print("Usage: python -m unlabeled_media_tagger.pipeline.annotate_image <image_path> [detector_backend]")
        print("\nDetector backends: retinaface (default), mtcnn, opencv, ssd, dlib, mediapipe")
        sys.exit(1)
    
    image_path = sys.argv[1]
    detector_backend = sys.argv[2] if len(sys.argv) > 2 else "retinaface"
    
    # Verify input image
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"Processing: {image_path}")
    print(f"Using detector: {detector_backend}\n")
    
    try:
        # Detect faces
        print("Detecting faces...")
        detections = detect_faces_in_image(image_path, detector_backend)
        print(f"Detected {len(detections)} face(s)\n")
        
        if len(detections) == 0:
            print("No faces detected. No annotation will be created.")
            sys.exit(0)
        
        # Display detection details
        for i, face in enumerate(detections, 1):
            bbox = face['bbox']
            confidence = face['confidence']
            print(f"Face {i}:")
            print(f"  Bounding Box: x={bbox['x']}, y={bbox['y']}, w={bbox['w']}, h={bbox['h']}")
            print(f"  Confidence: {confidence:.4f}")
        print()
        
        # Generate output path
        output_dir = Path("outputs")
        output_filename = f"annotated_{img_path.name}"
        output_path = output_dir / output_filename
        
        # Annotate image
        print("Annotating image...")
        annotate_image(image_path, detections, str(output_path))
        
        print(f"\n✓ Successfully annotated {len(detections)} face(s)")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
