# unlabeled-media-tagger

A modular pipeline for enriching unlabeled photos and videos in Google Drive using open-source computer vision models. It can download media from a Google Drive folder, detect and embed faces with DeepFace, cluster similar faces, write a face-clustering CSV, and optionally write compact clustering metadata back to each Drive file description.

## ✅ What Works Now

- **Face Detection**: Detect faces in images using DeepFace (RetinaFace backend)
- **Face Embeddings**: Generate face embeddings with DeepFace models such as ArcFace
- **Face Clustering**: Cluster similar face embeddings across downloaded media
- **Google Drive Media I/O**: List and download image/video files from a Drive folder
- **Pipeline CSV Output**: Write face clustering results to `outputs/pipeline/face_clusters.csv`
- **Google Drive Metadata Writeback**: Optionally update file descriptions with compact managed JSON metadata
- **Image Annotation**: Draw bounding boxes with confidence scores on detected faces
- **Video Annotation**: Sample video frames (1 fps) and annotate each frame with face detection results
- **Google Drive OAuth**: Local OAuth2 flow with token caching, file listing, and description updates
- **Debug Outputs**: Annotated images/frames saved to `outputs/` directory (for demo/verification; final pipeline will likely store structured detection results and only render annotations on demand)

## 🚀 Features (Planned)

- **Object Detection**: Identify objects and scenes in images
- **Named Identity Review**: Assign human-readable names to clusters
- **Local File Metadata Enrichment**: Write discovered tags and metadata to local media files
- **Better Cluster Review Tools**: Generate review sheets and approval workflows before Drive writeback

## 📁 Key Modules

**Working modules:**
- `src/unlabeled_media_tagger/pipeline/run.py` - End-to-end Drive download, extraction, detection, clustering, CSV output, and optional writeback
- `src/unlabeled_media_tagger/pipeline/fetch.py` - Fetch media files from Google Drive
- `src/unlabeled_media_tagger/pipeline/extract.py` - Extract image inputs and sampled video frames
- `src/unlabeled_media_tagger/pipeline/detect.py` - Run face embedding extraction for pipeline inputs
- `src/unlabeled_media_tagger/pipeline/compare.py` - Cluster face embeddings by cosine similarity
- `src/unlabeled_media_tagger/pipeline/enrich.py` - Format and write managed metadata to Drive descriptions
- `src/unlabeled_media_tagger/pipeline/detect_faces.py` - Face detection using DeepFace/RetinaFace
- `src/unlabeled_media_tagger/pipeline/embed_faces.py` - Face embedding extraction using DeepFace
- `src/unlabeled_media_tagger/pipeline/annotate_image.py` - Draw bounding boxes on images
- `src/unlabeled_media_tagger/pipeline/annotate_video.py` - Sample frames and annotate video
- `src/unlabeled_media_tagger/drive/auth.py` - Google Drive OAuth2 authentication
- `src/unlabeled_media_tagger/drive/files.py` - Drive file operations (list, download, get, update)
- `scripts/drive_smoke_test.py` - Drive integration smoke test

## 🔧 Installation

**Conda environment (recommended for the computer-vision pipeline):**

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

**Note**: TensorFlow and DeepFace are sensitive to version compatibility. The provided `environment.yml` contains the tested runtime configuration for the computer-vision pipeline.

**Development and testing dependencies:**

```bash
pip install -r requirements-dev.txt
```

## 📝 Usage

### Full Google Drive Pipeline

Process media files in a Google Drive folder and write clustering results to CSV:

```bash
python -m unlabeled_media_tagger "https://drive.google.com/drive/folders/YOUR_FOLDER_ID" \
  --output-dir outputs/pipeline \
  --recursive
```

The output CSV is written to:

```text
outputs/pipeline/face_clusters.csv
```

To also write compact clustering metadata back to each Google Drive file description:

```bash
python -m unlabeled_media_tagger "https://drive.google.com/drive/folders/YOUR_FOLDER_ID" \
  --output-dir outputs/pipeline \
  --recursive \
  --write-drive-descriptions
```

If you already ran the pipeline without writeback, update Drive descriptions later from the existing CSV:

```bash
python -m unlabeled_media_tagger \
  --writeback-from-csv outputs/pipeline/face_clusters.csv
```

### Reclustering From Cached Face Embeddings

Once detection and embedding are complete, you can experiment with clustering without
downloading media or rerunning DeepFace. This reads cached face records from
`outputs/pipeline/processed/` and writes a new CSV:

```bash
python -m unlabeled_media_tagger.pipeline.recluster \
  --processed-dir outputs/pipeline/processed \
  --csv-out outputs/recluster/face_clusters_reclustered.csv
```

The reclustering pass uses all cached embeddings at once. It considers high-similarity
face pairs first, then only merges clusters when the cross-cluster similarity quality
passes the configured gates. By default it also prevents one cluster from containing
two faces from the same sampled frame.

Useful reclustering options:

```bash
--edge-threshold 0.74          # Minimum pair similarity to consider a merge
--merge-min-similarity 0.62    # Lowest cross-pair similarity allowed in a merge
--merge-mean-similarity 0.70   # Average cross-cluster similarity required
--allow-same-frame-merge       # Disable same-frame cannot-link protection
```

For iterative review, run a reclustering experiment. This creates a dedicated
folder containing the reclustered CSV, a `manifest.json` with the threshold
settings, a summary CSV, contact sheets, and an `index.html` review page:

```bash
python scripts/run_recluster_experiment.py \
  --processed-dir outputs/pipeline/processed \
  --edge-threshold 0.74 \
  --merge-min-similarity 0.62 \
  --merge-mean-similarity 0.70
```

Outputs are written under:

```text
outputs/recluster_experiments/
```

Use `--name` to give an experiment a memorable folder name:

```bash
python scripts/run_recluster_experiment.py \
  --name stricter_same_frame_block \
  --edge-threshold 0.78 \
  --merge-min-similarity 0.66 \
  --merge-mean-similarity 0.74
```

Drive writeback preserves existing human description text and replaces only the managed block:

```text
<unlabeled-media-tagger>{"clusters":["person_000"],"face_count":1,...}</unlabeled-media-tagger>
```

Useful options:

```bash
--limit 10                    # Process at most 10 media files
--frame-interval 2.0          # Sample video frames every 2 seconds
--max-frames 50               # Limit sampled frames per video
--similarity-threshold 0.68   # Adjust face clustering strictness
--detector-backend retinaface # DeepFace detector backend
--model-name ArcFace          # DeepFace embedding model
--no-processing-cache         # Reprocess files even if cached face records exist
```

Processed face records are cached per Drive file under `outputs/pipeline/processed/`.
On reruns, cached files are skipped and their stored face records are reused for clustering.

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
# Install development dependencies, if needed
pip install -r requirements-dev.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=unlabeled_media_tagger
```

Current focused tests cover config defaults, pipeline stage contracts, clustering behavior, Drive metadata formatting, and file utility helpers.

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

## 🧾 Spreadsheet-Based Drive Writeback Starter

For a Colab workflow where a human updates confirmed names in Google Sheets and
then writes those names back to Google Drive file descriptions, see:

```text
examples/colab_drive_writeback_starter.py
```

The starter script assumes:

- Live Google Sheets access from Colab.
- A Face Library worksheet with `Face ID`, `Confirmed Name`, and `Confirm Status`.
- A Face Occurrences worksheet with `Face ID`, `Drive Link`, file metadata, and confidence columns.
- Only rows with a non-empty confirmed name and approved confirm status are written.
- Existing human-written Drive descriptions are preserved.
- Only the managed `<unlabeled-media-tagger>...</unlabeled-media-tagger>` block is replaced.
- `DRY_RUN = True` by default.


## 🛠️ Development

### Pipeline Stages

1. **Fetch Stage** (`pipeline/fetch.py`)
   - Authenticate with Google Drive API
   - Query and download media files
   - Manage local cache

2. **Extract Stage** (`pipeline/extract.py`)
   - Extract frames from videos at configurable intervals
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
   - Update Google Drive descriptions with a managed metadata block
   - Preserve existing human-written descriptions

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
