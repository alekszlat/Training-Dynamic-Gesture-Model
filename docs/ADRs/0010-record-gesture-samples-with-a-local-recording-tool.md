# ADR-0010: Record gesture samples with a local tool that hands over landmarks

|              |                |
| ------------ | -------------- |
| **Status**   | Proposed       |
| **Date**     | 2026-09-01     |
| **Deciders** | Hristo Hristov |

---

## Context and problem statement

The project needs recorded samples alongside the Jester data. `click` has no Jester equivalent, and the recorded set is what makes the model work on our own hands and cameras.

Collecting the footage by hand is slow. Recording clips, trimming them, sorting them into folders, then running `01_manifest_builder.py` and `02_landmark_extraction.py` over the whole set means a take is only judged hours or days later, when the lighting has changed and it can no longer be redone under the same conditions. All of it is mechanical work.

Both contributors need to record, not just the owner. That makes the handover artifact the deciding constraint: video is too large to pass around, and personal. Two things narrow the choice. `LabelEncoder` assigns class indices by position in `sorted(set(labels))`, so `y` means whatever the label set on the building machine says it means. And `03_build_tensors.py` applies the sequence length from ADR-0005, the feature layout from ADR-0006 and a split, which are the owner's decisions, not the recorder's.

How the dataset is finally composed and split is not decided here.

---

## Decision drivers

1. An unusable take must be caught while it can still be redone.
2. Contributors must be able to send samples without sending video.
3. The handover artifact must not encode the contributor's local settings.
4. Dataset and modelling decisions stay open for the owner.
5. No changes to the existing pipeline stages.

---

## Considered options

1. Record in bulk, let the batch pipeline surface unusable takes later.
2. Record take by take, hand over video files.
3. Record take by take, hand over landmark files and their metadata.
4. Record take by take, build tensors locally, hand over `train.pt` and `val.pt`.

---

## Decision

We will record take by take with a local tool that extracts landmarks immediately after each take, and hands over the landmark `.npy` files with a metadata csv in the format `03_build_tensors.py` already reads.

The landmarks extracted to judge a take are the same ones the pipeline wants, so nothing is read twice and stages 01 and 02 are not needed for recorded data.

Landmarks are also the last artifact before any modelling decision applies, and they are small enough to send. Tensors would fix the label encoding, sequence length, feature layout and split at the contributor's settings, and fail silently: `train.pt` carries no label mapping, because `TensorSaver.save_label_mapping` writes a separate `label_to_index.json` the owner's run overwrites. Six labels on one side against the five on the training pipeline branch gives four of five shared labels different indices, and the model trains normally while naming the wrong gesture.

**Takes go into `train`, `dev` and `eval` folders.** The tool has to write somewhere, and which takes were shot together is only knowable during the session. Recording that now keeps the option; pooling everything destroys it. Three folders because the set used during tuning and the set kept for a final score should differ.

**The quality bar is per session.** `min_frames` and `min_detection_rate` live in `Config` and are set before each session. A training session can be held to a clean bar; an eval session should look like deployment, where MediaPipe is noisier. The defaults are deliberately loose, because a real gesture is not steady enough for a high detection rate to mean quality rather than stillness.

**`sample_id` is built from the take's timestamped filename**, giving `recorded_click_20260901_143012_004821`. Nothing parses the id. It is an `.npy` filename, a print label, and a non-empty check in `manifest_combiner.py`, so only uniqueness matters. Counting rows in the local manifest, which is what `RecordedManifestBuilder` does, restarts at one on every machine, so two contributors would produce the same ids and silently overwrite each other's landmark files when their sets are merged. A timestamp is unique without anyone coordinating, which makes concatenating separately recorded sets a copy rather than a manual renumbering.

---

## Consequences

**Positive**

- An unusable take is redone in the same session.
- Contributors send landmarks, not video. A 120-frame take is about 10KB against megabytes as mp4.
- Contributions merge by concatenating a csv and copying `.npy` files.
- Nothing in the pipeline changes. The tool writes formats `SampleManifestReader` and `MetadataReader` already parse.

**Negative**

- Recorded samples now carry two id formats. The tool writes `recorded_click_20260901_143012_004821`, while `RecordedManifestBuilder` writes `recorded_000001`, so a corpus containing both is inconsistent to read and sort. Accepted because unordered unique ids merge without conflict and sequential ones do not.
- Landmarks cannot be re-extracted with different MediaPipe settings once the video is gone.
- The split lives only in the folder path. `Splitter` builds its own random split, so nothing downstream reads it.
- `RecordedManifestBuilder` treats every immediate subfolder as a label, so pointing `01_manifest_builder.py` at `data/raw/recorded` would read `train`, `dev` and `eval` as gesture names and emit nothing, without erroring.
- A rejected take is deleted, so the bar has to be chosen before the session, not after.
- The tool applies its bar independently of `LandmarkExtractionPipeline`, so the two can drift.

**Follow-on work**

- Agree the split and threshold questions with the owner, then make `Splitter` and tensor building read them. Owner: unassigned.

---

## Confirmation

`03_build_tensors.py` run against a session's `recorded_metadata.csv` produces `train.pt` with `x` of shape `[N, 40, 126]`, without 01 or 02 having run.

---

## More information

- [ADR-0005](0005-normalize-gesture-sequences-to-40-frames.md) and [ADR-0006](0006-use-wrist-relative-landmarks-and-frame-deltas.md) are the modelling decisions this handover keeps with the owner.
- [ADR-0007](0007-reject-low-detection-samples.md) sets the detection-rate rule. The tool applies it at record time, per session.
- `src/gesture_transformer/recording/README.md` documents usage and settings.
- Open, for the owner: how samples are assigned to splits, where the quality bar sits, and how a missed frame is represented. The last one matters before lowering any bar, because `LandmarkExtractor` writes a zero frame for an undetected one and `TemporalNormalizer` marks it valid, so the frame deltas turn it into a large false spike.
