# Evaluation protocol: cross-user robustness (Li and Hsieh method, one hand)

Status: agreed on 2026-11-13. Settings below are fixed before the test users are used.

## Method under test
- Landmarks: MediaPipe Hands, one hand, 21 key points.
- Features per frame (126): for each key point, position relative to the wrist (x, y, z) and change since the previous frame (dx, dy, dz). Based on Li and Hsieh (2025), 3D feature set, reduced to one hand (paper: 2 hands, 252).
- Clip length: up to 40 frames at 12 FPS (Jester frame rate). Shorter clips are zero-padded to 40. Padded frames are masked.
- Model: Transformer encoder, same settings for all runs.
- Classes (6): swipe up, swipe down, swipe left, swipe right, doing_other_things, no_gesture.

## Data
- Training and validation: Jester subset + recordings of one additional participant (12 FPS). Random 80/20 split of clips.
- Test: 5 new users, never used in training, validation, or tuning. Same camera and resolution as the additional participant, recorded at 12 FPS.
- Record all 6 classes for each test user. Aim for 20 or more clips per class per user (rule of thumb, not from a source).
- Write down for each test user: hand (left or right), distance to camera, light.
- Resample recordings to 12 FPS (nearest frame in time, or camera setting) with one method for every user.

## Training
- 5 seeds (1 to 5). Each seed makes a new random 80/20 split, new starting values, new clip order.
- 5 models in total. Save the split lists and the model files.
- Freeze all settings (learning rate, epochs rule, model size) before testing.

## Testing
- Test each of the 5 models once on the 5 unseen users. Do not change settings after this.

## Metrics
- Pooled macro-F1: put all clips of the 5 users together, then compute once (Forman and Scholz, 2010).
- Per-class precision and recall (pooled).
- Per-user accuracy (table).
- Confusion matrix (pooled).
- Processing time (MediaPipe time and model time, if measured).
- Report mean and standard deviation over the 5 seeds.

## Optional
- False-trigger rate: how often doing_other_things or no_gesture becomes a swipe.
- Landmark failure rate per user (clips where MediaPipe finds no hand).
- Speed and size test (play test clips faster or slower).

## Limits to state in the paper
- Validation shares users with training, so validation score is not a cross-user score.
- 5 test users: no significance test is possible. Describe trends only.
- Jester is low resolution (100 px, 12 FPS); the test recordings use a webcam.

## Sources
- Li and Hsieh (2025), Eng. Proc. 108, 22.
- Forman and Scholz (2010), Apples-to-Apples in Cross-Validation Studies, SIGKDD Explorations.
- Bouthillier et al. (2021), Accounting for Variance in ML Benchmarks, MLSys.
- Materzynska et al. (2019), The Jester Dataset, ICCVW.
