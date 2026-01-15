"""
Pipeline package containing modular stages for media processing.

Each stage module represents a distinct phase in the media enrichment pipeline:
- fetch: Retrieve media files from Google Drive
- extract: Extract frames, timestamps, and metadata from media
- detect: Run computer vision models for face and object detection
- compare: Compare and cluster detected faces across files
- enrich: Write discovered metadata back to source media
"""
