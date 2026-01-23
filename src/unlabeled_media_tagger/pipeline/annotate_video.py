"""
Video Face Annotation Module

This module provides functionality to annotate videos with face detection by sampling frames.
"""

import sys
import tempfile
from pathlib import Path
from typing import Optional
import cv2

from .detect_faces import detect_faces_in_image
from .annotate_image import annotate_image


def annotate_video(
    video_path: str,
    detector_backend: str = "retinaface",
    frame_interval_sec: float = 1.0
) -> None:
    """
    Process a video by sampling frames at specified intervals, detecting faces,
    and saving annotated frames.
    
    Args:
        video_path: Path to the input video file
        detector_backend: Detection backend to use (default: "retinaface")
                         Options: retinaface, mtcnn, opencv, ssd, dlib, mediapipe
        frame_interval_sec: Time interval between sampled frames in seconds (default: 1.0)
    
    Raises:
        FileNotFoundError: If video_path does not exist
        ValueError: If video cannot be opened or processed
    """
    # Verify video exists
    vid_path = Path(video_path)
    if not vid_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        
        print(f"Video properties:")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration_sec:.2f}s")
        print(f"  Sampling interval: {frame_interval_sec}s\n")
        
        # Create output directory
        video_stem = vid_path.stem
        output_dir = Path("outputs") / video_stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        processed_count = 0
        next_sample_time_ms = 0.0
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get current timestamp
            current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            
            # Check if we should sample this frame
            if current_time_ms >= next_sample_time_ms:
                processed_count += 1
                current_time_sec = current_time_ms / 1000.0
                
                print(f"Processing frame {frame_count} at t={current_time_sec:.1f}s")
                
                # Save frame to temporary file for detection
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                    cv2.imwrite(temp_path, frame)
                
                try:
                    # Detect faces in the frame
                    detections = detect_faces_in_image(temp_path, detector_backend)
                    print(f"  Detected {len(detections)} face(s)")
                    
                    # Generate output filename with frame number and timestamp
                    output_filename = f"frame_{frame_count:04d}_t{current_time_sec:.1f}s.jpg"
                    output_path = output_dir / output_filename
                    
                    # Annotate and save the frame
                    annotate_image(temp_path, detections, str(output_path))
                    
                finally:
                    # Clean up temporary file
                    Path(temp_path).unlink(missing_ok=True)
                
                # Calculate next sample time
                next_sample_time_ms += frame_interval_sec * 1000.0
            
            frame_count += 1
        
        print(f"\n✓ Processed {processed_count} frame(s)")
        print(f"✓ Annotated frames saved to: {output_dir}")
        
    finally:
        cap.release()


def main():
    """CLI entry point for video annotation."""
    if len(sys.argv) < 2:
        print("Usage: python -m unlabeled_media_tagger.pipeline.annotate_video <video_path> [detector_backend]")
        print("\nDetector backends: retinaface (default), mtcnn, opencv, ssd, dlib, mediapipe")
        sys.exit(1)
    
    video_path = sys.argv[1]
    detector_backend = sys.argv[2] if len(sys.argv) > 2 else "retinaface"
    
    # Verify input video
    vid_path = Path(video_path)
    if not vid_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    
    print(f"Processing video: {video_path}")
    print(f"Using detector: {detector_backend}\n")
    
    try:
        annotate_video(video_path, detector_backend)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
