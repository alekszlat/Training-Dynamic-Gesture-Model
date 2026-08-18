# ADR-0001: Use file-based contracts between dataset pipeline stages

|              |                    |
| ------------ | ------------------ |
| **Status**   | Accepted           |
| **Date**     | 2026-08-18         |
| **Deciders** | Alexander Zlatanov |

---

## Context and problem statement

The dataset preparation workflow has several stages: manifest creation, landmark extraction, and tensor building. Each stage may be rerun independently, and some stages, especially landmark extraction, are more expensive than others.

The project therefore needs a clear way for one stage to pass its results to the next without forcing the whole pipeline to run in one process.

---

## Considered options

1. Keep the entire dataset pipeline in one in-memory process.
2. Split the pipeline into stages that communicate through files.
3. Store intermediate state in a database.

---

## Decision

We will split the dataset pipeline into independent stages that communicate through files.

The main contracts are:

```text
01_manifest_builder.py
    -> data/manifests/samples.csv

02_landmark_extraction.py
    -> data/interim/landmarks/*.npy
    -> data/processed/metadata.csv

03_build_tensors.py
    -> data/processed/train.pt
    -> data/processed/val.pt
    -> data/processed/label_to_index.json
```

This keeps each stage independently runnable and allows intermediate results to be inspected and reused.

---

## Consequences

### Positive

- Expensive stages do not need to be repeated unnecessarily.
- Intermediate outputs can be inspected and validated.
- Each stage can be developed and tested separately.
- Data can be stored outside the source tree, including on external storage.

### Negative

- Intermediate files use additional disk space.
- File paths and formats become part of the pipeline contract.
- Outputs from different runs can become inconsistent if they are mixed.

---

## Confirmation

The decision is visible in the three top-level pipeline scripts and in the files they read and write.
