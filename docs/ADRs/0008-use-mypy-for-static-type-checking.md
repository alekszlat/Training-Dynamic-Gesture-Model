# ADR-0008: Use mypy for static type checking

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-08-12 |
| **Deciders** | Hristo Hristov |

---

## Context and problem statement

Issue #6 scoped a CI workflow but left the type checker itself undecided. The proposal just said "mypy or pyright" and nobody in the thread picked one. Once the lint and test jobs existed (`fad61e2`), the `typecheck` job needed an actual tool, not a placeholder.

The first attempt to run mypy against the codebase did not go cleanly. It reported the same classes as incompatible with themselves, which does not happen from a normal type error. The cause was a bug in how the project imported its own package, and it was inconsistent from file to file: some modules imported `gesture_transformer.*` directly, the installed package name, while others imported `src.gesture_transformer.*`, a longer path that also happened to resolve because of how the project sits on disk. Both forms ran without error, but a type checker treats two different import spellings as two different modules even when they point at the same file, so the same class existed twice as far as mypy was concerned and none of its comparisons made sense.

Fixed it by standardizing every import on the bare `gesture_transformer.*` form (`13cca6c`), the name the package is actually installed under. That same fix also removed the need for any separate pytest configuration: once every file settled on one spelling, pytest could rely on the editable install directly, with no conftest.py or pythonpath setting required to reconcile the mismatch. Only after that fix did mypy's output start reflecting actual problems in the code rather than an artifact of how it was imported.

Separately, `mediapipe`, a core dependency, ships no inline types and no stub package. Neither checker can see through it, so whichever one gets picked needs to be told to stop erroring on that import.

## Decision drivers

1. Has to work as a CI gate that fails a build, not just an editor hint.
2. Has to tolerate the project's src-layout and its one major untyped dependency without a rewrite.
3. Should be the tool most contributors already expect from a Python CI setup, so there is less to explain.

## Considered options

1. `mypy`
2. `pyright` (via its CLI, e.g. `uvx pyright` or the `pyright` npm package)

## Decision

We will use `mypy`, run as `uv run mypy src`. It is the reference implementation for PEP 484 and the default most people reach for in Python CI, which counts for more on a two-person project than a feature-by-feature comparison would.

Getting there required fixing the import bug described above and adding one override for `mediapipe`. The override does not solve the missing-types problem, it just tells mypy to stop treating that import as an error, so any call into mediapipe stays unchecked. That is a real limitation, not something this decision fixes, and it is spelled out below rather than glossed over.

## Consequences

**Positive**

- CI now fails on real type errors instead of only lint and formatting issues.
- Fixing the import bug was a precondition for mypy to say anything useful, and it also meant the test suite needed no separate pytest configuration, since every import settled on the one spelling pytest already resolves through the editable install.
- Config lives entirely in `pyproject.toml`, matching what issue #6 asked for: local and CI runs share the same settings.

**Negative**

- mypy is stricter about import resolution than pyright tends to be. The codebase only became checkable once the import bug was fixed. A checker that tolerated the mismatch quietly might have left it in place longer.
- `mediapipe` still is not type-checked. `[[tool.mypy.overrides]] module = "mediapipe.*"` with `ignore_missing_imports = true` stops mypy from erroring on the import, but everything past that boundary is treated as `Any`. Code like `mp.tasks.vision.HandLandmarker` in `landmark_extractor.py` gets no type verification at all. This is a suppression, not a fix, and it stays that way until mediapipe ships types or someone writes stubs for it.
- Contributors using Pylance in VS Code see pyright's opinions locally and mypy's in CI. The two do not always agree, so a clean editor does not guarantee a clean CI run.

**Follow-on work**

- None planned right now. Worth revisiting if mediapipe ever ships a `py.typed` marker or stubs, since the override could shrink or be dropped.

## Options in detail

### Option A: mypy

- **Good.** Reference implementation of PEP 484, the default assumption in most Python CI setups.
- **Good.** Configures entirely from `pyproject.toml`, alongside the rest of the project's tooling.
- **Bad.** Stricter about import roots. The pre-existing import bug had to be fixed first to get a real result instead of duplicate-type noise.
- **Neutral.** No built-in understanding of mediapipe. Needs an explicit override, which leaves that boundary unchecked.

### Option B: pyright

- **Good.** Powers Pylance, so it is already what contributors see live in VS Code. Using it in CI would mean local and CI feedback line up.
- **Good.** Infers a project's structure with less explicit config for common layouts.
- **Bad.** Built first as an editor language server, with the CLI as a secondary interface. Less the default assumption for a CI gate in most Python projects.
- **Neutral.** Same mediapipe gap as mypy. Neither tool can see into it.

## Confirmation

CI fails if `uv run mypy src` reports any error (`.github/workflows/ci.yml`, `typecheck` job).

## More information

- Issue #6, repository automation, left the type-checker choice open.
- Commit `13cca6c`, fixed the import bug that mypy's first run exposed, which also removed the need for any pytest path configuration.
- Commit `0309822`, added the `typecheck` CI job and the mediapipe override.
