# ADR-0002: Use a common frame reader interface for different data sources

|              |                    |
| ------------ | ------------------ |
| **Status**   | Accepted           |
| **Date**     | 2026-08-18         |
| **Deciders** | Alexander Zlatanov |

---

## Context and problem statement

The project uses gesture samples from different sources. Locally recorded samples are videos, while Jester samples are stored as frame sequences.

Landmark extraction should not need separate processing logic for every dataset source.

---

## Considered options

1. Add source-specific logic directly inside the landmark extraction pipeline.
2. Convert all datasets to one physical format before processing.
3. Use a common frame reader interface with one implementation per source type.

---

## Decision

We will use a common `FrameReader` interface for all data sources.

Each reader exposes the same operation:

```python
read_frames(path)
```

The current implementations are:

- `VideoReader` for recorded videos.
- `JesterFrameReader` for Jester frame folders.

`ReaderFactory` selects the correct reader from the sample's `source_type`.

---

## Consequences

### Positive

- Landmark extraction is independent of the original dataset format.
- Dataset-specific reading logic stays isolated.
- Adding another data source mainly requires a new reader and manifest builder.

### Negative

- Every supported data source must be adapted to the common frame abstraction.
- Source-specific metadata must be preserved separately if it is needed later.

---

## Confirmation

`FrameReader`, `VideoReader`, `JesterFrameReader`, and `ReaderFactory` implement this design in `src/gesture_transformer/datasets/readers/`.
