# Dynamic Gesture Recognition Model

Training pipeline for a dynamic hand-gesture recognition model using MediaPipe hand landmarks and a Transformer Encoder.
The project processes gesture recordings into normalized landmark tensors that can later be used for model training and evaluation.

## Current Status

The project is currently under development.

The intended workflow is:

```text
Dataset Preparation -> Training -> Evaluation
```

The current implementation primarily focuses on the dataset preparation pipeline.

## Pipeline

Dataset preparation is divided into three stages:

```text
01_manifest_builder.py -> 02_landmark_extraction.py -> 03_build_tensors.py
```

More detailed information about the pipeline, data contracts, and tensor shapes can be found in [`docs/`](docs/).

## Requirements

* Python 3.12
* `uv`
* MediaPipe
* OpenCV
* NumPy
* PyTorch

The complete dependency list is maintained in `pyproject.toml`.

## Setup

Clone the repository:

```bash
git clone <repository-url>
cd Training-Dynamic-Gesture-Model
```

Install the dependencies:

```bash
uv sync
```

## Data

The project expects datasets under the `data/` directory.

```text
data/
├── raw/            # Original datasets and recordings
├── manifests/      # Dataset manifests
├── interim/        # Intermediate processing results
└── processed/      # Metadata and generated training tensors
```

Large datasets and generated artifacts are not intended to be committed to the repository.

See the project documentation for more information about supported datasets and expected data structure.

## Running the Dataset Pipeline

### 1. Build the manifest

```bash
uv run 01_manifest_builder.py
```

Creates:

```text
data/manifests/samples.csv
```

### 2. Extract landmarks

```bash
uv run 02_landmark_extraction.py
```

Creates landmark data and processing metadata:

```text
data/interim/landmarks/
data/processed/metadata.csv
```

### 3. Build tensors

```bash
uv run 03_build_tensors.py
```

Creates the tensors used by the training pipeline:

```text
data/processed/train.pt
data/processed/val.pt
data/processed/label_to_index.json
```

## Development

Before committing changes, install the project's pre-commit hooks:

```bash
uv run pre-commit install
```

Run all configured checks manually with:

```bash
uv run pre-commit run --all-files
```

Pull requests are also checked through CI.

## Contributing

The project currently uses GitHub Issues, Pull Requests, and Trello to organize development.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development and contribution workflow.

## Documentation

Additional documentation is available in [`docs/`](docs/).

This includes information about:

* Project architecture
* Dataset and pipeline structure
* Data contracts
* Architectural decisions

Higher-level project goals and scope are documented in the project wiki.
