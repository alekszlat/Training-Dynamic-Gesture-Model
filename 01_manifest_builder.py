from config import (
    SAMPLES_DIR_EX_JESTER,
    SAMPLES_DIR_IN_JESTER,
    SAMPLES_DIR_RECORDED,
    SAMPLES_MANIFEST_PATH,
    SUPPORTED_LABELS,
)
from gesture_transformer.datasets.manifest.builders.jester_manifest_builder import (
    JesterManifestBuilder,
)
from gesture_transformer.datasets.manifest.builders.recorded_manifest_builder import (
    RecordedManifestBuilder,
)
from gesture_transformer.datasets.manifest.manifest_combiner import ManifestCombiner

if __name__ == "__main__":
    builder = RecordedManifestBuilder(SAMPLES_DIR_RECORDED)
    recorded_manifest = builder.build()

    builder = JesterManifestBuilder("local", SAMPLES_DIR_IN_JESTER)
    in_jester_manifest = builder.build()

    builder = JesterManifestBuilder("external", SAMPLES_DIR_EX_JESTER)
    ex_jester_manifest = builder.build()

    combiner = ManifestCombiner(
        [recorded_manifest, in_jester_manifest, ex_jester_manifest],
        SAMPLES_MANIFEST_PATH,
        SUPPORTED_LABELS,
    )
    if combiner.build_manifest():
        print(f"Combined manifest saved to {SAMPLES_MANIFEST_PATH}")
    else:
        print("Failed to build combined manifest due to invalid rows.")
