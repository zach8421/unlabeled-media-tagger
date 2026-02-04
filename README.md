# unlabeled-media-tagger

A modular pipeline for enriching unlabeled photos and videos in Google Drive using open-source computer vision models. Currently supports face detection with DeepFace/RetinaFace, visual annotation of images and videos, and basic Google Drive OAuth integration. Face comparison, clustering, and metadata writeback are planned.

## ✅ What Works Now

- **Face Detection**: Detect faces in images using DeepFace (RetinaFace backend)
- **Image Annotation**: Draw bounding boxes with confidence scores on detected faces
- **Video Annotation**: Sample video frames (1 fps) and annotate each frame with face detection results
- **Google Drive OAuth**: Local OAuth2 flow with token caching, file listing, and description updates
- **Debug Outputs**: Annotated images/frames saved to `outputs/` directory (for demo/verification; final pipeline will likely store structured detection results and only render annotations on demand)

## 🚀 Features (Planned)

- **Google Drive Media I/O**: Download/upload media bytes (currently only metadata operations supported)
- **Frame Extraction**: Extract frames at configurable intervals (currently fixed at 1 fps for video annotation)
- **Face Recognition / Embeddings**: Generate and compare face embeddings for identity matching
- **Object Detection**: Identify objects and scenes in images
- **Face Clustering**: Compare and cluster similar faces across your media collection
- **Metadata Enrichment**: Write discovered tags and metadata back to source files

## 📁 Key Modules

**Working modules:**
- `src/unlabeled_media_tagger/pipeline/detect_faces.py` - Face detection using DeepFace/RetinaFace
- `src/unlabeled_media_tagger/pipeline/annotate_image.py` - Draw bounding boxes on images
- `src/unlabeled_media_tagger/pipeline/annotate_video.py` - Sample frames and annotate video
- `src/unlabeled_media_tagger/drive/auth.py` - Google Drive OAuth2 authentication
- `src/unlabeled_media_tagger/drive/files.py` - Drive file operations (list, get, update)
- `scripts/drive_smoke_test.py` - Drive integration smoke test

**Planned (placeholders):**
- `src/unlabeled_media_tagger/pipeline/fetch.py`, `extract.py`, `detect.py`, `compare.py`, `enrich.py` - Full pipeline stages for clustering and metadata writeback

## 🔧 Installation

**Conda environment (recommended):**

```bash
# Clone the repository
git clone https://github.com/zach8421/unlabeled-media-tagger.git
cd unlabeled-media-tagger

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate unlabeled-media-tagger

# Install package in editable mode
pip install -e .
```

**Note**: TensorFlow and DeepFace are sensitive to version compatibility. The provided `environment.yml` contains the tested configuration.

## 📝 Usage

### Face Detection

```bash
# Detect faces in an image
python -m unlabeled_media_tagger.pipeline.detect_faces tests/assets/sample_image.jpg

# Annotate image with bounding boxes
python -m unlabeled_media_tagger.pipeline.annotate_image tests/assets/sample_image.jpg

# Annotate video (samples at 1 fps)
python -m unlabeled_media_tagger.pipeline.annotate_video tests/assets/sample_video.mpeg
```

**Output**: Annotated images/frames are saved to `outputs/` directory (debug/demo artifacts).

## 🧪 Testing

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=unlabeled_media_tagger
```

**Note**: Tests are currently minimal / in progress.

## 🧪 Drive Smoke Test

Test Google Drive OAuth and file operations:

**Setup:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create/select project, enable Google Drive API
3. Create OAuth 2.0 Client ID (Desktop app type)
4. Download credentials JSON and save as `secrets/credentials.json`

**Run:**
```bash
python scripts/drive_smoke_test.py
```

First run opens browser for OAuth consent. Token caches to `secrets/token.json`. By default, it updates the description of the first non-folder item in the folder.

**⚠️ Security**: Never commit `secrets/` directory (already in `.gitignore`)

**Scope used**: `https://www.googleapis.com/auth/drive` (full Drive access - prototype only)


## 🛠️ Development

### Pipeline Stages

1. **Fetch Stage** (`pipeline/fetch.py`)
   - Authenticate with Google Drive API
   - Query and download media files
   - Manage local cache

2. **Extract Stage** (`pipeline/extract.py`)
   - Extract frames from videos
   - Read EXIF metadata from images
   - Prepare data for analysis

3. **Detect Stage** (`pipeline/detect.py`)
   - Run facial detection models
   - Run object detection models
   - Generate facial embeddings

4. **Compare Stage** (`pipeline/compare.py`)
   - Compare facial embeddings
   - Cluster similar faces
   - Build face database

5. **Enrich Stage** (`pipeline/enrich.py`)
   - Format metadata
   - Write to local files
   - Update Google Drive metadata

### Future Integrations

- **Google Drive API**: OAuth2 authentication and file management
- **Computer Vision Models**: 
  - Face detection (e.g., MTCNN, RetinaFace)
  - Face recognition (e.g., FaceNet, ArcFace)
  - Object detection (e.g., YOLO, Faster R-CNN)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📚 Documentation

Detailed documentation is coming soon in the `docs/` directory.
