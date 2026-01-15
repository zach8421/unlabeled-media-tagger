# unlabeled-media-tagger

A modular pipeline for enriching unlabeled photos and videos in Google Drive using open-source computer vision and facial recognition models. The system detects faces and visual features, extracts frames and timestamps, compares faces across files, and writes discovered metadata back to the source media.

## 🚀 Features (Planned)

- **Google Drive Integration**: Fetch and update media files directly from Google Drive
- **Frame Extraction**: Extract frames from videos at configurable intervals
- **Facial Detection & Recognition**: Detect faces and generate embeddings for comparison
- **Object Detection**: Identify objects and scenes in images
- **Face Clustering**: Compare and cluster similar faces across your media collection
- **Metadata Enrichment**: Write discovered tags and metadata back to source files

## 📁 Project Structure

```
unlabeled-media-tagger/
├── src/
│   └── unlabeled_media_tagger/
│       ├── __init__.py
│       ├── __main__.py           # Main entry point
│       ├── pipeline/              # Pipeline stage modules
│       │   ├── __init__.py
│       │   ├── fetch.py          # Google Drive media retrieval
│       │   ├── extract.py        # Frame and metadata extraction
│       │   ├── detect.py         # Computer vision models
│       │   ├── compare.py        # Face comparison and clustering
│       │   └── enrich.py         # Metadata writeback
│       ├── config/                # Configuration management
│       │   ├── __init__.py
│       │   └── settings.py       # Configuration classes
│       └── utils/                 # Utility functions
│           ├── __init__.py
│           ├── logging.py        # Logging utilities
│           └── file_utils.py     # File system utilities
├── tests/                         # Test suite
│   ├── pipeline/
│   ├── config/
│   └── utils/
├── docs/                          # Documentation
├── examples/                      # Example configurations and usage
│   ├── config.example.json
│   ├── config.example.yaml
│   └── example_pipeline.py
├── pyproject.toml                 # Project configuration
├── requirements.txt               # Core dependencies
├── requirements-dev.txt           # Development dependencies
└── README.md                      # This file
```

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/zach8421/unlabeled-media-tagger.git
cd unlabeled-media-tagger

# Install the package in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## 📝 Usage

### Basic Usage

```python
from unlabeled_media_tagger.pipeline.fetch import FetchStage
from unlabeled_media_tagger.pipeline.extract import ExtractStage
from unlabeled_media_tagger.pipeline.detect import DetectStage
from unlabeled_media_tagger.pipeline.compare import CompareStage
from unlabeled_media_tagger.pipeline.enrich import EnrichStage
from unlabeled_media_tagger.config.settings import Config

# Initialize configuration
config = Config()

# Initialize and run pipeline stages
# (Implementation pending)
```

See `examples/example_pipeline.py` for a more complete example.

### Configuration

Copy one of the example configuration files and customize for your setup:

```bash
cp examples/config.example.yaml config.yaml
# Edit config.yaml with your settings
```

## 🧪 Testing

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=unlabeled_media_tagger
```

## 🛠️ Development

### Project Status

⚠️ **This project is currently in the initial skeleton phase.** The structure and modules are in place, but implementation is pending.

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
