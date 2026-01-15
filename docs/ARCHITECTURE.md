# Architecture Overview

## System Design

The unlabeled-media-tagger is designed as a modular pipeline with distinct stages that can be developed and tested independently.

## Pipeline Stages

### 1. Fetch Stage
**Purpose**: Retrieve media files from Google Drive  
**Inputs**: Google Drive credentials, folder ID, query filters  
**Outputs**: List of downloaded media files with metadata  
**Future Dependencies**: google-api-python-client, google-auth

### 2. Extract Stage
**Purpose**: Extract frames and metadata from media files  
**Inputs**: Media file paths  
**Outputs**: Extracted frames, timestamps, EXIF data  
**Future Dependencies**: opencv-python, pillow, exifread

### 3. Detect Stage
**Purpose**: Run computer vision models for detection  
**Inputs**: Image frames  
**Outputs**: Face bounding boxes, embeddings, object labels  
**Future Dependencies**: torch, facenet-pytorch, opencv-python

### 4. Compare Stage
**Purpose**: Compare and cluster detected faces  
**Inputs**: Face embeddings from multiple files  
**Outputs**: Clustered faces, unique person IDs  
**Future Dependencies**: scikit-learn, numpy

### 5. Enrich Stage
**Purpose**: Write metadata back to source files  
**Inputs**: Detected metadata, file references  
**Outputs**: Updated files with enriched metadata  
**Future Dependencies**: google-api-python-client, exiftool

## Data Flow

```
Google Drive → Fetch → Extract → Detect → Compare → Enrich → Google Drive
                 ↓        ↓        ↓         ↓         ↓
              [Cache] [Frames] [Faces]  [Clusters] [Metadata]
```

## Configuration

Configuration is managed through the `config.settings` module, which provides:
- Google Drive API settings
- Model paths and parameters
- Pipeline processing options
- Output directories and formats

## Extensibility

The modular design allows for:
- Easy addition of new detection models
- Pluggable storage backends (not just Google Drive)
- Custom metadata formats
- Alternative clustering algorithms
- Additional pipeline stages

## Future Considerations

- Batch processing optimization
- Distributed processing support
- Caching strategies
- Error handling and recovery
- Progress tracking and logging
- API/web interface
