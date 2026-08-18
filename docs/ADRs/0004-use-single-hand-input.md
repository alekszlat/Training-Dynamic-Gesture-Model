# ADR-0004: Use single-hand input for gesture recognition

|              |                    |
| ------------ | ------------------ |
| **Status**   | Accepted           |
| **Date**     | 2026-08-18         |
| **Deciders** | Alexander Zlatanov |

---

## Context and problem statement

The source paper uses a broader hand representation, but the application that will consume this model is designed around single-hand interaction.

Supporting two hands would increase the input representation and require decisions about hand ordering, missing hands, and two-handed gestures.

---

## Considered options

1. Support two hands as in the broader paper setup.
2. Use one detected hand because that matches the consuming application.

---

## Decision

We will represent each gesture using one detected hand.

MediaPipe Hand Landmarker is configured with:

```text
num_hands = 1
```

Each frame therefore contains 21 landmarks with three coordinates each before feature construction.

---

## Consequences

### Positive

- The model input is smaller and simpler.
- The preprocessing pipeline matches the intended consuming application.
- There is no ambiguity about ordering two detected hands.

### Negative

- Two-handed gestures cannot be represented correctly.
- The project is an adaptation rather than an exact reproduction of the source paper.
- Adding two-hand support later would change preprocessing and model input dimensions.

---

## Confirmation

`LandmarkExtractor` defaults to `num_hands=1` and uses the first detected hand.
