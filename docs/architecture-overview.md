# Training-Dynamic-Gesture-Model Architecture Overview

|                   |                    |
| ----------------- | ------------------ |
| **Owner**         | Alexander Zlatanov |
| **Reviewer**      | Hristo Hristov     |
| **Last reviewed** | 2026-08-14         |

---

## 1. Purpose and scope

This project builds and evaluates a model for recognising dynamic hand gestures using MediaPipe hand landmarks and a Transformer-based model. It handles the processing of gesture data, model training, and evaluation. The trained model is intended to be used by another application.

Inside the project are dataset preparation, landmark extraction, tensor creation, training, and evaluation.

Outside the project are the user interface, live camera handling, MediaPipe itself, and the application logic that uses the recognised gesture.

---

## 2. Constraints

- The model operates on a single detected hand.
- Hand landmarks are extracted using MediaPipe Hand Landmarker.
- Each hand is represented by 21 landmarks with `(x, y, z)` coordinates.
- Input gesture sequences are normalised to a fixed length of 40 frames.
- The project is implemented in Python and uses PyTorch for model development and training.
- Dataset stages communicate through files rather than directly calling the next stage.
- The project uses both locally recorded gesture samples and selected samples from the Jester dataset.
- Only the gestures in the current pipeline configuration are included in the
  prepared dataset.

The current label set is a development/testing configuration used to exercise
the data pipeline and Transformer training workflow. The final production
gesture vocabulary has not yet been selected and may change as the application
requirements are finalized.

Current development/testing gesture labels are:

- `swiping_left`
- `swiping_right`
- `swiping_up`
- `swiping_down`
- `click`
- `doing_other_things`
- `no_gesture`

---

## 3. System context

The project receives gesture samples from external datasets or locally recorded videos, processes them into training-ready tensors, and produces a trained gesture-recognition model.

```text

Data sources → Training system → Model artifact → Consuming application

```
