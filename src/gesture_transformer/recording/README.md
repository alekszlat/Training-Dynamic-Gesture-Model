# Recording Tool

Records gesture samples one take at a time instead of collecting footage in bulk and
processing it later. You pick a label and a split, record a take from the webcam, and the
tool writes it to `output_dir/<split>/<label>/<label>_<timestamp>.mp4`. Each take is then
checked with MediaPipe before it is kept, so a sample where the hand was missed is caught
straight away rather than hours later during the batch pipeline.

## Usage

TODO: add the commands to run once the session runner exists.

## Settings

All settings live in `Config` in `config.py`.

| Setting | Meaning |
|---|---|
| `output_dir` | Root folder for takes. |
| `active_split` | Split being recorded into. |
| `active_label` | Gesture being recorded. Required. |
| `min_frames` | Fewest frames a take may have. |
| `min_detection_rate` | Lowest share of frames with a hand, 0 to 1. |
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

Splits are kept apart on disk and in separate metadata files, so a sample cannot leak from
one into another. Record whole sessions into one split at a time, because takes from the
same session look alike and splitting them apart inflates your scores.
