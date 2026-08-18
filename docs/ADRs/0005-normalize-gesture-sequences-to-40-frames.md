# ADR-0005: Normalize gesture sequences to 40 frames using sampling and padding

|              |                    |
| ------------ | ------------------ |
| **Status**   | Accepted           |
| **Date**     | 2026-08-18         |
| **Deciders** | Alexander Zlatanov |

---

## Context and problem statement

Gesture samples contain different numbers of frames. The model requires batches with a consistent sequence length, so variable-length landmark sequences must be converted to a fixed size.

The project uses 40 frames as the current target sequence length.

---

## Considered options

1. Keep variable-length sequences.
2. Truncate all long sequences from one side and pad short ones.
3. Uniformly sample long sequences and zero-pad short sequences to a fixed length.

---

## Decision

We will normalize every gesture sequence to 40 frames.

The current strategy is:

```text
T = 40  -> keep the sequence unchanged
T > 40  -> uniformly sample 40 frames across the sequence
T < 40  -> keep the original frames and zero-pad the remainder
T = 0   -> produce an all-zero sequence
```

A mask marks which frames contain original data and which are padding.

The exact value of 40 is part of the current model contract. Its rationale should not be described as experimentally optimal unless that is later demonstrated.

---

## Consequences

### Positive

- Every sample has the same temporal dimension.
- Samples can be stacked into batches.
- Uniform sampling keeps information from across the whole gesture.
- The mask allows padded frames to be distinguished from real frames.

### Negative

- Long gestures lose frames.
- Short gestures contain artificial zero frames.
- Temporal speed and fine-grained motion can be altered by resampling.
- Changing the target length later changes the model input contract.

---

## Confirmation

`TemporalNormalizer` implements the 40-frame normalization strategy and produces both a normalized sequence and a mask.
