# ADR-0007: Reject samples with insufficient hand detection

|              |                    |
| ------------ | ------------------ |
| **Status**   | Accepted           |
| **Date**     | 2026-08-18         |
| **Deciders** | Alexander Zlatanov |

---

## Context and problem statement

MediaPipe does not detect a hand in every frame of every sample. Keeping every sample regardless of detection quality would allow heavily incomplete landmark sequences into training.

At the same time, rejecting a sample because of one or two missed frames would discard otherwise usable data.

---

## Considered options

1. Reject any sample containing a missed detection.
2. Keep every sample regardless of detection rate.
3. Preserve occasional missing frames but reject samples below a minimum detection rate.

---

## Decision

We will preserve missing-detection frames as zero landmark frames and reject a sample when fewer than 70% of its frames contain detected hand landmarks.

The detection rate is:

```text
detected_frames / total_frames
```

The current minimum accepted value is:

```text
0.7
```

The threshold is a current modelling assumption and should not be described as experimentally optimal unless later evaluation supports that claim.

---

## Consequences

### Positive

- Occasional MediaPipe misses do not automatically discard a sample.
- The temporal position of frames is preserved.
- Very low-quality samples are prevented from reaching tensor building and training.

### Negative

- Accepted samples can still contain artificial zero frames.
- Some potentially useful samples are discarded.
- The 0.7 threshold may need to be revisited after dataset analysis or model evaluation.

---

## Confirmation

`LandmarkExtractor` inserts zero landmark frames when no hand is detected, and `LandmarkExtractionPipeline` rejects samples whose `detection_rate` is below `0.7`.
