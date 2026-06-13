"""Extract Stage - media frame and metadata extraction."""

from pathlib import Path

from unlabeled_media_tagger.utils.file_utils import is_image_file, is_video_file


class UnreadableMediaError(ValueError):
    """A media file cannot be decoded — unsupported type, or a corrupt/
    unopenable video. This is a *permanent* property of the file (a successful
    re-download won't fix it), so callers can treat it as terminal rather than
    retrying. Subclasses ValueError for backward compatibility with callers
    that catch the broader type.
    """


class ExtractStage:
    """
    Extract stage for processing media files and extracting relevant data.
    
    This stage is responsible for:
    - Extracting frames from videos at specified intervals
    - Extracting timestamps and duration information
    - Reading EXIF data from images
    - Preparing media data for analysis
    """
    
    def __init__(self, config=None):
        """
        Initialize the extract stage.
        
        Args:
            config: Configuration dictionary for extraction settings
        """
        self.config = config or {}
    
    def extract(self, media_file):
        """
        Extract frames and metadata from a media file.
        
        Args:
            media_file: Path to the media file to process
            
        Returns:
            Dictionary containing extracted frames, timestamps, and metadata
            
        Raises:
            FileNotFoundError: If media_file does not exist.
            ValueError: If the file type is unsupported or a video cannot be read.
        """
        media_path = Path(media_file)
        if not media_path.exists():
            raise FileNotFoundError(f"Media file not found: {media_file}")

        if is_image_file(str(media_path)):
            return {
                "source_path": str(media_path),
                "media_type": "image",
                "frames": [
                    {
                        "path": str(media_path),
                        "timestamp_sec": None,
                        "frame_index": None,
                    }
                ],
                "metadata": {},
            }

        if is_video_file(str(media_path)):
            return self._extract_video_frames(media_path)

        raise UnreadableMediaError(f"Unsupported media file type: {media_file}")

    def _extract_video_frames(self, media_path: Path) -> dict:
        import cv2

        frame_interval = float(self.config.get("frame_interval", 1.0))
        max_frames = int(self.config.get("max_frames", 100))
        output_dir = Path(self.config.get("frame_dir", "outputs/frames")) / media_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(media_path))
        if not cap.isOpened():
            raise UnreadableMediaError(f"Failed to open video: {media_path}")

        frames = []
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps if fps > 0 else 0
            next_sample_time_ms = 0.0
            frame_index = 0

            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if current_time_ms >= next_sample_time_ms:
                    timestamp_sec = current_time_ms / 1000.0
                    frame_path = output_dir / f"frame_{frame_index:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frames.append(
                        {
                            "path": str(frame_path),
                            "timestamp_sec": timestamp_sec,
                            "frame_index": frame_index,
                        }
                    )
                    next_sample_time_ms += frame_interval * 1000.0

                frame_index += 1
        finally:
            cap.release()

        return {
            "source_path": str(media_path),
            "media_type": "video",
            "frames": frames,
            "metadata": {
                "fps": fps,
                "total_frames": total_frames,
                "duration_sec": duration_sec,
            },
        }
