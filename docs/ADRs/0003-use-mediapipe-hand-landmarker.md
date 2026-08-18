# ADR-0003: Use MediaPipe Hand Landmarker for landmark extraction

|              |                    |
| ------------ | ------------------ |
| **Status**   | Accepted           |
| **Date**     | 2026-08-18         |
| **Deciders** | Alexander Zlatanov |

---

## Context and problem statement

The project trains its own dynamic gesture-recognition model. The model therefore needs hand landmark sequences as input rather than only a final gesture label from another recognizer.

MediaPipe provides both higher-level gesture recognition functionality and lower-level hand landmark detection.

---

## Considered options

1. Use MediaPipe's higher-level gesture recognition as the final recognizer.
2. Use MediaPipe Hand Landmarker only for landmark extraction and train the project's own model.

---

## Decision

We will use MediaPipe Hand Landmarker to extract hand landmarks and perform gesture classification with the project's own model.

For each detected hand, MediaPipe provides 21 landmarks with `(x, y, z)` coordinates.

---

## Consequences

### Positive

- The project keeps control over the dynamic gesture model.
- Landmark sequences can be processed and transformed before training.
- The approach supports adapting the research-paper pipeline.

### Negative

- The project must implement and maintain its own training and evaluation pipeline.
- Model quality depends on the quality of MediaPipe landmark detection.
- MediaPipe becomes a required preprocessing dependency.

---

## Confirmation

`LandmarkExtractor` uses `mp.tasks.vision.HandLandmarker` and returns landmark arrays with shape `[T, 21, 3]`.
