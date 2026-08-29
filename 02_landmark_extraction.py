from config import (
    LANDMARK_DELEGATE,
    LANDMARK_OUTPUT_DIR,
    MANIFEST_PATH,
    MEDIAPIPE_MODEL_PATH,
    METADATA_OUTPUT_PATH,
)
from gesture_transformer.datasets.landmark_extraction.landmark_extraction_pipeline import (
    LandmarkExtractionPipeline,
)
from gesture_transformer.datasets.landmark_extraction.landmark_extractor import (
    LandmarkExtractor,
)
from gesture_transformer.datasets.landmark_extraction.landmark_saver import (
    LandmarkSaver,
)
from gesture_transformer.datasets.landmark_extraction.metadata_writer import (
    MetadataWriter,
)
from gesture_transformer.datasets.manifest.sample_manifest_reader import (
    SampleManifestReader,
)
from gesture_transformer.datasets.readers.reader_factory import ReaderFactory

if __name__ == "__main__":
    manifest_reader = SampleManifestReader(manifest_path=MANIFEST_PATH)
    reader_factory = ReaderFactory()
    landmark_extractor = LandmarkExtractor(
        model_path=MEDIAPIPE_MODEL_PATH,  # gitleaks:allow
        delegate=LANDMARK_DELEGATE,
    )
    landmark_saver = LandmarkSaver(landmarks_dir=LANDMARK_OUTPUT_DIR)
    metadata_writer = MetadataWriter(output_path=METADATA_OUTPUT_PATH)

    pipeline = LandmarkExtractionPipeline(
        manifest_reader=manifest_reader,
        reader_factory=reader_factory,
        landmark_extractor=landmark_extractor,
        landmark_saver=landmark_saver,
        metadata_writer=metadata_writer,
        min_detection_rate=0.7,
    )

    pipeline.run()
