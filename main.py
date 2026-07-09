from src.gesture_transformer.datasets.manifest.recorded_manifest_builder import RecordedManifestBuilder
from src.gesture_transformer.datasets.manifest.jester_manifest_builder import JesterManifestBuilder
from pathlib import Path
    

if __name__ == "__main__":

    #samples_dir = Path("data/raw/recorded")
    #builder = RecordedManifestBuilder(samples_dir)
    #manifest = builder.build()

    samples_dir = Path("data/raw/jester/20bn-jester-v1")
    builder = JesterManifestBuilder(samples_dir)
    manifest = builder.build()

    for sample in manifest:
        print(sample)

