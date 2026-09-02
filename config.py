"""Central configuration for the gesture-recognition data and training pipeline.

The numbered root scripts use this module as the composition/configuration layer:

01_manifest_builder.py
    Builds the combined sample manifest.

02_landmark_extraction.py
    Extracts MediaPipe hand landmarks from source samples.

03_build_tensors.py
    Converts extracted landmarks into training-ready PyTorch tensors.

04_training_T_encoder.py
    Trains and validates the Transformer gesture classifier.
"""

from pathlib import Path

import mediapipe as mp
import torch

# ---------------------------------------------------------------------------
# Shared project paths
# ---------------------------------------------------------------------------

# Absolute path to the repository root.
# Using PROJECT_ROOT prevents scripts from depending on the current
# working directory from which they are executed.
PROJECT_ROOT = Path(__file__).resolve().parent

# Directory containing processed pipeline artifacts such as:
# metadata.csv, train.pt, val.pt, and label_to_index.json.
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# 01_manifest_builder.py
# ---------------------------------------------------------------------------

# Directory containing personally recorded gesture videos.
# Expected structure:
#
# data/raw/recorded/
# ├── swiping_left/
# ├── swiping_right/
# ├── swiping_up/
# ├── swiping_down/
# └── click/
SAMPLES_DIR_RECORDED = PROJECT_ROOT / "data" / "raw" / "recorded"

# Root directory of the Jester dataset used by the project.
SAMPLES_DIR_JESTER = PROJECT_ROOT / "data" / "raw" / "jester" / "20bn-jester-v1"

# Combined manifest produced from the recorded and Jester datasets.
# This file becomes the input contract for landmark extraction.
SAMPLES_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "samples.csv"

# Canonical internal gesture classes supported by this project.
#
# These labels should be the single source of truth for deciding which
# samples are allowed into the dataset pipeline.
SUPPORTED_LABELS = {
    "swiping_left",
    "swiping_right",
    "swiping_up",
    "swiping_down",
    "click",
    "doing_other_things",
    "no_gesture",
}


# ---------------------------------------------------------------------------
# 02_landmark_extraction.py
# ---------------------------------------------------------------------------

# Manifest containing all samples that should pass through landmark extraction.
MANIFEST_PATH = SAMPLES_MANIFEST_PATH

# MediaPipe Hand Landmarker model used to extract 21 hand landmarks per frame.
MEDIAPIPE_MODEL_PATH = (
    PROJECT_ROOT / "src" / "gesture_transformer" / "models" / "hand_landmarker.task"
)

# Directory where extracted landmark arrays are stored.
# Each sample is saved as a NumPy file before tensor construction.
LANDMARK_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "landmarks"

# Metadata generated during landmark extraction.
# Contains information such as frame count, detection rate,
# extraction status, and landmark-file location.
METADATA_OUTPUT_PATH = PROCESSED_DIR / "metadata.csv"

# MediaPipe inference backend.
#
# GPU uses the MediaPipe GPU delegate for Hand Landmarker inference.
# Change to BaseOptions.Delegate.CPU if GPU execution is unavailable
# or causes compatibility problems on the target machine.
LANDMARK_DELEGATE = mp.tasks.BaseOptions.Delegate.GPU


# ---------------------------------------------------------------------------
# 03_build_tensors.py
# ---------------------------------------------------------------------------

# Metadata produced by the landmark-extraction stage and consumed by
# the tensor-building pipeline.
METADATA_PATH = METADATA_OUTPUT_PATH

# Fixed number of temporal positions used for each gesture sequence.
# Shorter gestures are padded and longer sequences are normalized
# to this length.
TARGET_SEQUENCE_LENGTH = 40


# ---------------------------------------------------------------------------
# 04_training_T_encoder.py
# ---------------------------------------------------------------------------

# Directory for generated model checkpoints.
# Training artifacts are kept under outputs/ rather than src/.
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

# Filename used for the checkpoint with the lowest validation loss.
BEST_MODEL_FILENAME = "best_model.pth"

# Automatically use CUDA when supported by the current PyTorch installation.
# Otherwise, training falls back to CPU.
TRAINING_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tensor artifacts produced by 03_build_tensors.py.
TRAIN_FILE = "train.pt"
VAL_FILE = "val.pt"

# Label-to-index mapping generated during tensor construction.
# Training uses this artifact to determine the model's number of classes
# instead of hard-coding num_classes.
LABEL_MAPPING_FILE = PROCESSED_DIR / "label_to_index.json"


# ---------------------------------------------------------------------------
# DataLoader configuration
# ---------------------------------------------------------------------------

# Number of worker processes used by each PyTorch DataLoader.
#
# 0 = load data in the main process.
# Values > 0 allow background loading but introduce multiprocessing overhead.
NUM_WORKERS = 1

# Number of gesture samples processed in one optimizer step.
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Trainer configuration
# ---------------------------------------------------------------------------

# Maximum number of complete passes through the training dataset.
NUM_EPOCHS = 30


# ---------------------------------------------------------------------------
# Optimizer configuration
# ---------------------------------------------------------------------------

# Initial learning rate used by AdamW.
LEARNING_RATE = 5e-4


# ---------------------------------------------------------------------------
# ReduceLROnPlateau configuration
# ---------------------------------------------------------------------------

# Validation loss is minimized, therefore lower values represent improvement.
SCHEDULER_MODE = "min"

# Multiply the current learning rate by this value when a plateau is detected.
# Example:
# 0.0005 -> 0.00025
SCHEDULER_FACTOR = 5e-1

# Number of non-improving epochs tolerated before reducing the learning rate.
SCHEDULER_PATIENCE = 2

# Lower bound below which the scheduler cannot reduce the learning rate.
MIN_LEARNING_RATE = 1e-6
