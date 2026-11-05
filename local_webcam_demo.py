"""
Classifies a gesture recorded from the webcam, using the demo checkpoint.

Separate from the numbered pipeline scripts. A clip goes through the same chain
training used, MediaPipe landmarks into TemporalNormalizer into FeatureBuilder
into the model, so what the demo shows is what the model actually saw.

Run with:

    uv run python local_webcam_demo.py

Space starts a two second capture, space again stops it early, q quits. The 21
landmarks are drawn on every frame so you can see whether the hand is tracked
before committing to a take.

Reads the checkpoint written by local_train_demo.py.

Dependencies:
    gesture_transformer.datasets: landmark extraction and feature building.
    gesture_transformer.training: the model the checkpoint restores into.

Author:
    Hristo Hristov
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from config import LANDMARK_DELEGATE
from gesture_transformer.datasets.landmark_extraction.landmark_extractor import (
    LandmarkExtractor,
)
from gesture_transformer.datasets.tensor_extraction.feature_builder import (
    FeatureBuilder,
)
from gesture_transformer.datasets.tensor_extraction.temporal_normalizer import (
    TemporalNormalizer,
)
from gesture_transformer.training.gesture_transformer_en import GestureTransformer

CAMERA_INDEX = 0
RECORD_SECONDS = 2.0
OVERLAY_SECONDS = 3.0
WINDOW_NAME = "gesture demo"

DEMO_DIR = Path("outputs/local_demo")
CHECKPOINT_PATH = DEMO_DIR / "best_model.pth"
MODEL_ARGS_PATH = DEMO_DIR / "model_args.json"
LABEL_MAP_PATH = DEMO_DIR / "processed" / "label_to_index.json"
LANDMARKER_PATH = Path("src/gesture_transformer/models/hand_landmarker.task")

# MediaPipe 0.10.x no longer ships mp.solutions.hands.HAND_CONNECTIONS.
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def draw_hud(frame: np.ndarray, lines: list[str], color: tuple[int, int, int]) -> None:
    """
    Draw the status lines in the top-left corner of a frame.

    Args:
        frame: BGR frame, written in place.
        lines: Text to stack downwards, one line each.
        color: BGR colour for the text.

    Returns:
        None.
    """

    y = 34
    for line in lines:
        cv2.putText(
            frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )
        y += 32


def draw_hand_landmarks(frame: np.ndarray, landmarks: np.ndarray) -> None:
    """
    Draw one hand's 21 landmarks and the bones between them onto a frame.

    Args:
        frame: BGR frame, written in place.
        landmarks: Landmarks for one hand, normalized to 0 to 1.

    Returns:
        None.
    """

    landmarks = np.asarray(landmarks)
    if landmarks.size == 0 or not np.any(landmarks):
        return

    height, width = frame.shape[:2]
    points = [
        (
            int(np.clip(float(x), 0.0, 1.0) * (width - 1)),
            int(np.clip(float(y), 0.0, 1.0) * (height - 1)),
        )
        for x, y, _z in landmarks[:21]
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, (0, 255, 255), -1, cv2.LINE_AA)


def draw_bars(frame: np.ndarray, probabilities: list[float], labels: list[str]) -> None:
    """
    Draw one horizontal probability bar per class in the bottom-left corner.

    Args:
        frame: BGR frame, written in place.
        probabilities: One probability per class, in label order.
        labels: Class names, ordered by class index.

    Returns:
        None.
    """

    height = frame.shape[0]
    top = height - 20 - len(labels) * 26

    for i, (label, probability) in enumerate(zip(labels, probabilities)):
        y = top + i * 26
        width = int(probability * 240)
        cv2.rectangle(frame, (150, y - 14), (150 + 240, y + 4), (60, 60, 60), -1)
        cv2.rectangle(frame, (150, y - 14), (150 + width, y + 4), (0, 200, 255), -1)
        cv2.putText(
            frame,
            f"{label[:14]:<14}",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{probability * 100:4.0f}%",
            (400, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )


def classify_clip(
    frames: list[np.ndarray],
    extractor: LandmarkExtractor,
    normalizer: TemporalNormalizer,
    feature_builder: FeatureBuilder,
    model: GestureTransformer,
) -> np.ndarray | None:
    """
    Run one recorded clip through the full chain and classify it.

    Args:
        frames: BGR frames captured for this take.
        extractor: MediaPipe landmark extractor, already open.
        normalizer: Pads or samples the clip to the model's sequence length.
        feature_builder: Turns landmarks into the model's input features.
        model: Model to classify with, already in eval mode.

    Returns:
        One probability per class, or None if no hand was detected.
    """

    result = extractor.extract(frames)
    print(
        f"[diag] {result.total_frames} frames, "
        f"hand detected in {result.detected_frames}/{result.total_frames}"
    )

    if result.landmarks is None or len(result.landmarks) == 0:
        return None

    normalized = normalizer.normalize(np.asarray(result.landmarks))
    features = feature_builder.build(normalized.norm_sequence, normalized.mask)

    x = torch.tensor(np.asarray(features), dtype=torch.float32).unsqueeze(0)

    # The model wants PyTorch padding semantics: True marks padded positions.
    padding_mask = torch.tensor(np.asarray(normalized.mask)).unsqueeze(0) == 0

    with torch.no_grad():
        logits = model(x, padding_mask)

    return torch.softmax(logits, dim=1)[0].numpy()


def main() -> None:
    """
    Open the webcam and classify gestures until the user quits.

    Returns:
        None.
    """

    with MODEL_ARGS_PATH.open(encoding="utf-8") as file:
        model_args = json.load(file)

    with LABEL_MAP_PATH.open(encoding="utf-8") as file:
        label_to_index = json.load(file)

    labels = [
        label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])
    ]

    model = GestureTransformer(**model_args)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"[diag] checkpoint from epoch {checkpoint['epoch']}, "
        f"val_accuracy {checkpoint['val_accuracy']:.4f}"
    )

    extractor = LandmarkExtractor(
        model_path=LANDMARKER_PATH, delegate=LANDMARK_DELEGATE
    )
    normalizer = TemporalNormalizer(model_args["max_len"])
    feature_builder = FeatureBuilder()

    capture = cv2.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam (index {CAMERA_INDEX}).")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

    state = "idle"
    recorded: list[np.ndarray] = []
    record_start = 0.0
    overlay_lines: list[str] = []
    overlay_probabilities: np.ndarray | None = None
    overlay_until = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Failed to read from webcam.")
                break

            now = time.time()
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            if state == "recording":
                recorded.append(frame.copy())
                remaining = RECORD_SECONDS - (now - record_start)
                draw_hud(
                    frame,
                    [
                        f"RECORDING {remaining:0.1f}s",
                        f"{len(recorded)} frames",
                        "space = stop now",
                    ],
                    (0, 0, 255),
                )
                cv2.circle(frame, (frame.shape[1] - 30, 30), 12, (0, 0, 255), -1)

                if remaining <= 0 or key == ord(" "):
                    state = "classifying"
            else:
                lines = ["space = record ~2s gesture", "q = quit"]
                if now < overlay_until:
                    lines = overlay_lines + lines
                else:
                    overlay_probabilities = None
                draw_hud(frame, lines, (0, 255, 0))

                if overlay_probabilities is not None:
                    draw_bars(frame, list(overlay_probabilities), labels)

                if key == ord(" "):
                    state = "recording"
                    recorded = []
                    record_start = time.time()

            live = extractor.extract([frame])
            if live.landmarks is not None and len(live.landmarks) > 0:
                draw_hand_landmarks(frame, live.landmarks[0])

            cv2.imshow(WINDOW_NAME, frame)

            if state == "classifying":
                probabilities = classify_clip(
                    recorded, extractor, normalizer, feature_builder, model
                )

                if probabilities is None:
                    overlay_lines = ["No hand detected"]
                    overlay_probabilities = None
                else:
                    best = int(np.argmax(probabilities))
                    overlay_lines = [
                        f"{labels[best]}  {probabilities[best] * 100:.1f}%"
                    ]
                    overlay_probabilities = probabilities
                    print(
                        "[diag] "
                        + ", ".join(
                            f"{label}={probability * 100:.1f}%"
                            for label, probability in zip(labels, probabilities)
                        )
                    )

                overlay_until = time.time() + OVERLAY_SECONDS
                recorded = []
                state = "idle"
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
