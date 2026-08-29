from config import (
    OUTPUTH_PATH_SAMPLES_CSV,
    SAMPLES_DIR_JESTER,
    SAMPLES_DIR_RECORDED,
    SUPPORTED_LABELS,
)
from gesture_transformer.datasets.manifest.jester_manifest_builder import (
    JesterManifestBuilder,
)
from gesture_transformer.datasets.manifest.manifest_combiner import ManifestCombiner
from gesture_transformer.datasets.manifest.recorded_manifest_builder import (
    RecordedManifestBuilder,
)

if __name__ == "__main__":
    builder = RecordedManifestBuilder(SAMPLES_DIR_RECORDED)
    recorded_manifest = builder.build()

    builder = JesterManifestBuilder(SAMPLES_DIR_JESTER, SUPPORTED_LABELS)
    jester_manifest = builder.build()

    combiner = ManifestCombiner(
        recorded_manifest, jester_manifest, OUTPUTH_PATH_SAMPLES_CSV, SUPPORTED_LABELS
    )
    if combiner.build_manifest():
        print(f"Combined manifest saved to {OUTPUTH_PATH_SAMPLES_CSV}")
    else:
        print("Failed to build combined manifest due to invalid rows.")
