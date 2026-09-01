# Recording Tool

Records gesture samples one take at a time and turns them into trainable data in the
same pass, instead of collecting footage in bulk and processing it later. You pick a
label and a split, record a take from the webcam, and the tool writes it to
`output_dir/<split>/<label>/<label>_<timestamp>.mp4`. Each take is then checked with
MediaPipe before it is kept, so a sample where the hand was missed is caught straight
away rather than hours later during the batch pipeline.

## Usage

Set `LABEL` and `SPLIT` at the top of `record_session.py`, then run:

```bash
uv run python record_session.py
```

During a session:

| Key | Does |
|---|---|
| `space` | Start the next take. |
| `q` | Stop and save the current take. |
| `e` | End the session and write its takes out. |

Each finished take is checked straight away and its result is printed, along with how
many takes the session has kept so far. Rejected takes are deleted, so the folders and
the files stay in step.

`min_detection_rate` is set per session, not once. A training session wants a clean bar,
an evaluation session should look like deployment, where MediaPipe is noisier. Choose it
before the session starts, because a rejected take is gone.

Ending a session writes three things, all in the formats the batch pipeline already
uses, and all appended so earlier sessions stay where they are:

| Output | Contents |
|---|---|
| `landmarks_dir` | One `.npy` of landmarks per take, `[frames, 21, 3]`. |
| `metadata_path` | Landmark metadata, what `03_build_tensors.py` reads. |
| `manifest_path` | The take list, what `02_landmark_extraction.py` reads. |

The landmarks come from the check the validator already ran, so a take is never read
twice. That means `03_build_tensors.py` can run straight off a session without going
through 01 or 02, and the videos never have to leave your machine.

Recording needs `src/gesture_transformer/models/hand_landmarker.task`, which is
gitignored and downloaded separately, same as the batch pipeline.

## Settings

All settings live in `Config` in `config.py`.

| Setting | Meaning |
|---|---|
| `output_dir` | Root folder for takes. |
| `active_split` | Split being recorded into. |
| `manifest_path` | Manifest takes are appended to. |
| `metadata_path` | Landmark metadata takes are appended to. |
| `landmarks_dir` | Folder for the extracted landmark files. |
| `active_label` | Gesture being recorded. Required. |
| `min_frames` | Fewest frames a take may have. |
| `min_detection_rate` | Lowest share of frames with a hand, 0 to 1. |
| `model_path` | MediaPipe model used to check takes. |
| `webcam_id` | Which camera to open. |
| `default_fps` | Fallback when the camera reports nonsense. |
| `max_fps` | Highest frame rate accepted. |
| `max_consecutive_read_failures` | Failed reads in a row before a take ends. |

## Dataset splits

Every take is recorded into exactly one split, chosen by `active_split`.

| Split | Purpose |
|---|---|
| `train` | The model learns from these. |
| `validation` | Tuning and picking the best checkpoint. Never trained on. |
| `test` | Final score only. Touched once, at the end. |

The split lives in the folder path only. It is not a column in either file, and
`03_build_tensors.py` makes its own random train and validation split, so nothing
downstream knows about it yet. Record whole sessions into one split at a time, because
takes from the same session look alike and splitting them apart inflates your scores.
