from pathlib import Path

from src.gesture_transformer.datasets.manifest.jester_manifest_builder import (
    JesterManifestBuilder,
)
from src.gesture_transformer.datasets.manifest.manifest_combiner import ManifestCombiner
from src.gesture_transformer.datasets.manifest.recorded_manifest_builder import (
    RecordedManifestBuilder,
)

if __name__ == "__main__":
    samples_dir = Path("data/raw/recorded")
    builder = RecordedManifestBuilder(samples_dir)
    recorded_manifest = builder.build()

    samples_dir = Path("data/raw/jester/20bn-jester-v1")
    builder = JesterManifestBuilder(samples_dir)
    jester_manifest = builder.build()

    output_path = Path("data/manifests/samples.csv")

    combiner = ManifestCombiner(recorded_manifest, jester_manifest, output_path)
    if combiner.build_manifest():
        print(f"Combined manifest saved to {output_path}")
    else:
        print("Failed to build combined manifest due to invalid rows.")
