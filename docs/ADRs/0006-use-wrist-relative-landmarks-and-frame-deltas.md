# ADR-0006: Use wrist-relative landmarks and frame deltas as model features

|              |                    |
| ------------ | ------------------ |
| **Status**   | Accepted           |
| **Date**     | 2026-08-18         |
| **Deciders** | Alexander Zlatanov |

---

## Context and problem statement

Raw MediaPipe landmarks contain absolute image-space positions. A dynamic gesture recognizer needs information about both hand pose and movement over time.

Using only raw coordinates would leave the model to learn both translation handling and temporal differences from the original landmark values.

---

## Considered options

1. Use raw landmark coordinates only.
2. Use wrist-relative coordinates only.
3. Use wrist-relative coordinates together with frame-to-frame deltas.

---

## Decision

We will use wrist-relative landmark coordinates together with frame-to-frame landmark deltas.

For each frame:

```text
21 landmarks x 3 wrist-relative coordinates = 63 values
21 landmarks x 3 frame-delta coordinates    = 63 values
                                                   ----
Total                                         = 126 values
```

The resulting feature sequence has shape:

```text
[40, 126]
```

No additional scale normalization is currently applied.

---

## Consequences

### Positive

- Absolute hand translation in the image is reduced as a factor.
- The model receives explicit motion information.
- Both hand pose and frame-to-frame movement are represented.

### Negative

- The feature dimension doubles from 63 to 126.
- Hand scale and camera distance are still represented because scale normalization is not applied.
- Delta values depend on the temporal sampling strategy.

---

## Confirmation

`FeatureBuilder` subtracts the wrist position, calculates frame-to-frame deltas, concatenates both representations, and flattens them to 126 features per frame.
