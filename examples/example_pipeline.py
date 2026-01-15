"""
Example usage of the unlabeled-media-tagger pipeline.

This example demonstrates how to use the pipeline stages once they are implemented.
"""

from unlabeled_media_tagger.pipeline.fetch import FetchStage
from unlabeled_media_tagger.pipeline.extract import ExtractStage
from unlabeled_media_tagger.pipeline.detect import DetectStage
from unlabeled_media_tagger.pipeline.compare import CompareStage
from unlabeled_media_tagger.pipeline.enrich import EnrichStage
from unlabeled_media_tagger.config.settings import Config


def run_pipeline():
    """
    Example pipeline execution flow.
    
    This is a skeleton showing how the stages will be connected.
    Each stage is currently not implemented and will raise NotImplementedError.
    """
    # Initialize configuration
    config = Config()
    
    # Initialize pipeline stages
    fetch_stage = FetchStage(config=config.google_drive)
    extract_stage = ExtractStage(config=config.pipeline)
    detect_stage = DetectStage(config=config.models)
    compare_stage = CompareStage(config=config.pipeline)
    enrich_stage = EnrichStage(config=config.pipeline)
    
    print("Pipeline stages initialized successfully!")
    print("Stage 1: Fetch - Retrieve media from Google Drive")
    print("Stage 2: Extract - Extract frames and metadata")
    print("Stage 3: Detect - Run computer vision models")
    print("Stage 4: Compare - Compare and cluster faces")
    print("Stage 5: Enrich - Write metadata back to media")
    
    # TODO: Implement pipeline execution
    # media_files = fetch_stage.fetch()
    # for media_file in media_files:
    #     extracted_data = extract_stage.extract(media_file)
    #     faces = detect_stage.detect_faces(extracted_data['frames'])
    #     objects = detect_stage.detect_objects(extracted_data['frames'])
    #     clusters = compare_stage.compare_faces(faces)
    #     enrich_stage.enrich_drive(media_file['id'], clusters)


if __name__ == "__main__":
    run_pipeline()
