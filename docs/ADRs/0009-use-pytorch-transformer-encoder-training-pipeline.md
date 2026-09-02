# ADR-0009: Use a PyTorch Transformer encoder training pipeline

|              |                     |
| ------------ | ------------------- |
| **Status**   | Accepted            |
| **Date**     | 2026-08-28          |
| **Deciders** | Project maintainers |

---

## Context and problem statement

The preprocessing pipeline produces training-ready PyTorch `.pt` files in `data/processed/`.

The current training pipeline loads:

- `train.pt`
- `val.pt`

Each processed dataset is expected to contain:

- `x`
- `y`
- `mask`
- `sample_ids`
- `labels`

The current one-hand feature representation contains 126 values per frame:

`21 landmarks × 6 values = 126 features`

The six values per landmark are:

- `x`
- `y`
- `z`
- `Δx`
- `Δy`
- `Δz`

Gesture sequences are currently normalized to 40 frames by the tensor-building
configuration.

The current processed development dataset therefore follows:

`[N, 40, 126]`

where:

- `N` = number of samples
- `40` = current configured sequence length
- `126` = current features per frame

The model must classify each sequence into one of the classes represented in
the current dataset label mapping.

The current label set is a development/testing configuration used to exercise
the data pipeline and Transformer training workflow. The final production
gesture vocabulary has not yet been selected and may change as the application
requirements are finalized.

Current development/testing gesture labels are:

- `swiping_left`
- `swiping_right`
- `swiping_up`
- `swiping_down`
- `click`
- `doing_other_things`
- `no_gesture`

The tensor-building entry point receives `TARGET_SEQUENCE_LENGTH` from
`config.py`. The training entry point derives:

- `input_dim` from `train_data["x"].shape[-1]`
- `max_len` from `train_data["x"].shape[-2]`
- `num_classes` from `label_to_index.json`

The classifier output size follows the generated dataset label mapping rather
than a hard-coded production gesture vocabulary.

Shorter sequences contain padded positions. These positions must not influence
Transformer attention or final sequence pooling.

The training pipeline must also support:

- batching and shuffling
- CPU/CUDA execution
- training and validation
- CrossEntropyLoss
- AdamW
- ReduceLROnPlateau
- best-model checkpointing

---

## Decision

The project will use a PyTorch training pipeline composed of:

```text
.pt files
    ↓
GestureDatasetLoader
    ↓
GestureDataset
    ↓
DataLoader
    ↓
GestureTransformer
    ↓
CrossEntropyLoss
    ↓
AdamW
    ↓
Trainer
```

The model receives both features and a padding mask.

The padding mask is used in:

1. Transformer attention
2. masked mean pooling

Training and validation are handled by a separate `Trainer` class.

Internal Python imports use the installed package path:

```text
gesture_transformer.*
```

`src.gesture_transformer.*` is not a supported import path.

---

## Dataset loading

`GestureDatasetLoader` loads processed `.pt` files from:

`data/processed/`

The current files are:

- `train.pt`
- `val.pt`

Files are loaded with:

```python
torch.load(
    file_path,
    map_location="cpu",
    weights_only=True,
)
```

The loader verifies that:

- the data directory exists
- the requested file exists
- the loaded object is a dictionary
- the dictionary is not empty

---

## Dataset contract

`GestureDataset` expects the keys:

```text
x
y
mask
sample_ids
labels
```

They are exposed internally as:

```text
x          -> features
y          -> answers
mask       -> padding_mask
sample_ids -> sample_ids
labels     -> labels
```

Expected shapes:

```text
features:
[N, T, F]
torch.float32

answers:
[N]
torch.int64 / torch.long

padding_mask:
[N, T]
torch.bool after Dataset conversion

sample_ids:
list[str]

labels:
list[str]
```

The full padding-mask shape must match `features.shape[:2]`.

A single dataset sample contains:

```text
features:
[T, F]

answers:
scalar class index

padding_mask:
[T]

sample_id:
string

label:
string
```

The sample type is mixed: tensors are returned for `features`, `answers`, and
`padding_mask`; strings are returned for `sample_id` and `label`.

---

## Padding-mask semantics

The processed mask uses:

```text
1 = valid frame
0 = padded frame
```

PyTorch Transformer padding masks use:

```text
False = valid frame
True  = padded frame
```

The Dataset converts the mask with:

```python
self.padding_mask = data[padding_mask] == 0
```

Example:

```text
Stored mask:
1 1 1 1 0 0

Converted mask:
F F F F T T
```

The stored mask is used instead of deriving padding from zero-valued feature
vectors because a valid frame may still contain an all-zero feature vector.

---

## DataLoader

DataLoaders are created using `create_data_loader`.

Current configuration values come from `config.py`:

```text
batch_size = config.BATCH_SIZE
num_workers = config.NUM_WORKERS
```

Training loader:

```python
shuffle=True
```

Validation loader:

```python
shuffle=False
```

A normal batch contains:

```text
features:
[B, T, F]

answers:
[B]

padding_mask:
[B, T]
```

The DataLoader also batches `sample_id` and `label` metadata.

---

## Device handling

The training script receives its device from `config.TRAINING_DEVICE`, which is
defined as:

```python
TRAINING_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

The model is constructed with dimensions derived from the processed training
artifacts and then moved to that device:

```python
model = GestureTransformer(
    input_dim=input_dim,
    max_len=sequence_length,
    num_classes=num_classes,
).to(TRAINING_DEVICE)
```

Inside training and validation, these tensors are moved to the selected device:

```python
features = batch["features"].to(self.device)
padding_mask = batch["padding_mask"].to(self.device)
answers = batch["answers"].to(self.device)
```

Metadata such as `sample_id` and `label` remains on CPU.

The positional encoding moves with the model because it is registered as a buffer.

---

## Target validation

The training entry point validates target class indices before training.

Both training and validation targets must be in:

```text
[0, num_classes - 1]
```

The pipeline fails before training if either split contains an empty target
tensor or a target index outside that range.

The current entry point also checks that every class in `label_to_index.json`
has at least one training sample.

---

## Model architecture

`GestureTransformer` requires these dimensions explicitly:

```text
input_dim        = train_data["x"].shape[-1]
max_len          = train_data["x"].shape[-2]
num_classes      = len(label_to_index)
```

The current architecture hyperparameters are:

```text
hidden_dim       = 64
num_layers       = 4
num_heads        = 8
dim_feedforward  = 256
dropout          = 0.2
```

The model pipeline is:

```text
[B, T, input_dim]
      ↓
Linear(input_dim, hidden_dim)
      ↓
[B, T, hidden_dim]
      ↓
Sinusoidal positional encoding
      ↓
[B, T, hidden_dim]
      ↓
4 × Transformer encoder layers
      ↓
[B, T, hidden_dim]
      ↓
Masked mean pooling
      ↓
[B, hidden_dim]
      ↓
Linear(hidden_dim, num_classes)
      ↓
[B, num_classes] logits
```

---

## Linear projection

The model projects each frame from `input_dim` features to `hidden_dim`
features:

```python
self.projection = nn.Linear(
    input_dim,
    hidden_dim,
)
```

This changes:

```text
[B, T, input_dim]
```

to:

```text
[B, T, hidden_dim]
```

---

## Positional encoding

The model uses fixed sinusoidal positional encoding.

The encoding:

- uses sine values for even dimensions
- uses cosine values for odd dimensions
- has shape `[1, max_len, hidden_dim]`
- is added to the projected features

It is registered with:

```python
self.register_buffer("pe", pe)
```

and retrieved with:

```python
self.get_buffer("pe")
```

It is part of model state but is not trainable.

---

## Transformer encoder

The encoder layer is configured as:

```python
nn.TransformerEncoderLayer(
    d_model=hidden_dim,
    nhead=num_heads,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    batch_first=True,
)
```

The model stacks four encoder layers with:

```python
nn.TransformerEncoder(
    encoder_layer=encoder_layer,
    num_layers=num_layers,
)
```

The padding mask is passed through:

```python
src_key_padding_mask=padding_mask
```

The Transformer preserves shape:

```text
[B, T, hidden_dim]
→
[B, T, hidden_dim]
```

The model `forward` contract requires:

```text
x:
[B, T, input_dim]

padding_mask:
bool [B, T]
False = valid position
True  = padded position
```

The model validates that:

- `padding_mask` has dtype `torch.bool`
- `padding_mask.shape == x.shape[:2]`
- each sample contains at least one valid frame

---

## Masked mean pooling

The Transformer returns:

```text
[B, T, hidden_dim]
```

Padded encoder outputs are explicitly overwritten before pooling:

```python
x = x.masked_fill(
    padding_mask.unsqueeze(-1),
    0.0,
)
```

This uses `masked_fill` instead of multiplication because:

```text
NaN * 0 = NaN
```

Overwriting padded positions guarantees that padded NaN values are removed
before pooling. After padded positions are cleared, the model checks the
remaining encoder output with:

```python
torch.isfinite(x).all()
```

Any remaining NaN or Inf belongs to a valid sequence position and is treated as
a real numerical error.

The number of valid frames is calculated from the inverted padding mask:

```python
valid_frame_count = (~padding_mask).sum(dim=1, keepdim=True).to(x.dtype)
```

The final sequence representation is:

```python
x = x.sum(dim=1) / valid_frame_count
```

This reduces:

```text
[B, T, hidden_dim]
```

to:

```text
[B, hidden_dim]
```

Only valid frames contribute to the mean.

All-padded samples are rejected before pooling, so the valid-frame count is
never zero.

Masked mean pooling is a project implementation choice.

---

## Classifier

The classifier is:

```python
self.classifier = nn.Linear(
    hidden_dim,
    num_classes,
)
```

`num_classes` is derived from `label_to_index.json` for the current dataset
build, so this output size follows the active label mapping rather than a
hard-coded production gesture vocabulary.

The model returns raw logits:

```text
[B, num_classes]
```

Softmax is not applied in `forward`.

---

## Loss

Training uses:

```python
nn.CrossEntropyLoss()
```

The loss is calculated outside the model:

```python
loss = criterion(
    logits,
    answers,
)
```

Raw logits are passed directly to CrossEntropyLoss.

---

## Optimizer

Training uses AdamW:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)
```

Current learning rate:

```text
0.0005
```

The optimizer is created outside the model and injected into the Trainer.

---

## Scheduler

Training uses:

```python
torch.optim.lr_scheduler.ReduceLROnPlateau
```

Current scheduler configuration:

```text
mode     = "min"
factor   = 0.5
patience = 2
min_lr   = 1e-6
```

The scheduler monitors validation loss:

```python
self.scheduler.step(val_loss)
```

The current learning rate is read from:

```python
self.optimizer.param_groups[0]["lr"]
```

and printed after every epoch.

---

## Trainer

The `Trainer` class stores and uses:

- device
- checkpoint directory
- checkpoint filename
- model
- training DataLoader
- validation DataLoader
- optimizer
- criterion
- scheduler

It contains:

```text
train_one_epoch()
validate_one_epoch()
train()
```

---

## Training epoch

Training begins with:

```python
self.model.train()
```

For each batch:

```text
move tensors to device
        ↓
optimizer.zero_grad()
        ↓
forward pass
        ↓
CrossEntropyLoss
        ↓
loss.backward()
        ↓
optimizer.step()
```

Training loss is accumulated using the actual batch size:

```python
total_loss += loss.item() * features.size(0)
```

Average epoch loss:

```python
avg_loss = total_loss / total_samples
```

Predictions are calculated using:

```python
logits.argmax(dim=1)
```

Training accuracy is:

```python
total_correct / total_samples
```

---

## Validation epoch

Validation begins with:

```python
self.model.eval()
```

Gradient tracking is disabled:

```python
with torch.no_grad():
```

Validation does not call:

```text
loss.backward()
optimizer.step()
```

Validation loss is also sample-weighted:

```python
total_loss += loss.item() * features.size(0)
```

Validation accuracy is calculated using the argmax class predictions.

---

## Training history

`Trainer.train()` records and returns per-epoch history:

```text
epoch
train_loss
train_accuracy
val_loss
val_accuracy
learning_rate
```

The current implementation keeps this history in memory and returns it from
`train()`. It does not yet save the history to disk.

---

## Training duration

The training duration is configured in `config.py`:

```python
NUM_EPOCHS = 30
```

The training entry point passes that value to the Trainer:

```python
trainer.train(NUM_EPOCHS)
```

---

## Best-model checkpointing

Checkpoints are saved under the configured checkpoint directory:

```text
outputs/checkpoints/
```

The current checkpoint filename is:

```text
best_model.pth
```

The directory is created automatically with:

```python
checkpoint_dir.mkdir(
    parents=True,
    exist_ok=True,
)
```

The initial best validation loss is:

```python
best_val_loss = float("inf")
```

The checkpoint is saved only when:

```python
val_loss < best_val_loss
```

Validation loss is therefore the model-selection criterion.

---

## Checkpoint contents

The saved checkpoint contains:

```python
{
    "epoch": epoch + 1,
    "model_state_dict": self.model.state_dict(),
    "optimizer_state_dict": self.optimizer.state_dict(),
    "scheduler_state_dict": self.scheduler.state_dict(),
    "val_loss": val_loss,
    "val_accuracy": val_accuracy,
}
```

This preserves:

- model parameters
- optimizer state
- scheduler state
- epoch
- validation loss
- validation accuracy

Checkpoint loading and resumed training are not yet implemented.

---

## Current configuration

The numbered root scripts use `config.py` as the composition/configuration
layer.

| Parameter              | Current source/value                                      |
| ---------------------- | --------------------------------------------------------- |
| Processed directory    | `config.PROCESSED_DIR` -> `data/processed`                |
| Training file          | `config.TRAIN_FILE` -> `train.pt`                         |
| Validation file        | `config.VAL_FILE` -> `val.pt`                             |
| Label mapping file     | `config.LABEL_MAPPING_FILE` -> `label_to_index.json`      |
| Checkpoint directory   | `config.CHECKPOINT_DIR` -> `outputs/checkpoints`          |
| Checkpoint filename    | `config.BEST_MODEL_FILENAME` -> `best_model.pth`          |
| Input dimension        | Derived from processed training tensor shape              |
| Sequence length        | Derived from processed training tensor shape              |
| Tensor target length   | `config.TARGET_SEQUENCE_LENGTH` -> `40`                   |
| Hidden dimension       | `64`                                                      |
| Transformer layers     | `4`                                                       |
| Attention heads        | `8`                                                       |
| Feed-forward dimension | `256`                                                     |
| Dropout                | `0.2`                                                     |
| Number of classes      | Derived from `label_to_index.json`                        |
| Batch size             | `config.BATCH_SIZE` -> `32`                               |
| DataLoader workers     | `config.NUM_WORKERS` -> `1`                               |
| Epochs                 | `config.NUM_EPOCHS` -> `30`                               |
| Loss                   | `CrossEntropyLoss`                                        |
| Optimizer              | `AdamW`                                                   |
| Initial learning rate  | `config.LEARNING_RATE` -> `0.0005`                        |
| Scheduler              | `ReduceLROnPlateau`                                       |
| Scheduler mode         | `config.SCHEDULER_MODE` -> `min`                          |
| Scheduler factor       | `config.SCHEDULER_FACTOR` -> `0.5`                        |
| Scheduler patience     | `config.SCHEDULER_PATIENCE` -> `2`                        |
| Minimum learning rate  | `config.MIN_LEARNING_RATE` -> `0.000001`                  |
| Device                 | `config.TRAINING_DEVICE`: CUDA when available, else CPU   |

---

## Consequences

### Positive

- Uses standard PyTorch abstractions.
- Dataset loading and model training are separated.
- Required dataset structure is validated before training.
- The explicit preprocessing mask is preserved.
- Padding-mask semantics are converted once at the Dataset boundary.
- Valid zero-valued frames are not automatically treated as padding.
- Padding is ignored during Transformer attention.
- Padding is ignored during mean pooling.
- The model returns raw logits compatible with CrossEntropyLoss.
- Model, loss, optimizer, and Trainer responsibilities are separated.
- Training and validation are separate.
- Validation uses no gradient computation.
- Epoch losses account for partial final batches.
- CUDA is used automatically when available.
- Positional encoding follows device placement automatically.
- Validation loss controls learning-rate scheduling.
- Validation loss controls best-model checkpointing.
- Later epochs with worse validation loss do not overwrite the best model.
- Optimizer and scheduler states are preserved for future resume support.
- Input dimension, sequence length, and class count are derived from generated
  dataset artifacts.

### Negative

- The pipeline is coupled to the `.pt` keys `x`, `y`, `mask`, `sample_ids`, and `labels`.
- Hyperparameters are module-level constants rather than a dedicated configuration object.
- Only the best validation-loss checkpoint is kept.
- Checkpoint restoration is not yet implemented.
- Resumed training is not yet implemented.
- Training history is returned but not yet saved to disk.
- Evaluation currently provides only loss and accuracy.
- Confusion matrix is not yet implemented.
- Precision, recall, and F1 are not yet implemented.
- Untouched test-set evaluation is not yet implemented.

---

## Follow-on work

- Add automated tests for `GestureDataset` loading and validation.
- Add automated tests for mask conversion and mask-shape validation.
- Add model output-shape tests.
- Add masked mean-pooling tests.
- Add a padded-NaN test.
- Add an all-padded sample rejection test.
- Add Trainer training and validation tests.
- Add scheduler stepping tests.
- Add device handling tests.
- Add checkpoint-saving tests.
- Save per-epoch training history to disk.
- Plot training and validation loss.
- Plot training and validation accuracy.
- Add confusion matrix.
- Add precision.
- Add recall.
- Add F1 score.
- Verify train/validation split for data leakage.
- Evaluate on an untouched test set.
- Add checkpoint loading.
- Add standalone inference.
- Add resumed-training support.

---

## Confirmation

The current implementation includes:

### `GestureDatasetLoader`

Loads processed `.pt` dictionaries from CPU storage.

### `GestureDataset`

Validates dataset structure, converts the stored mask into PyTorch padding-mask
semantics, and exposes individual gesture samples with mixed tensor/string
types.

### `create_data_loader`

Creates training and validation DataLoaders.

### `SinusoidalPositionalEncoding`

Creates fixed temporal positional encodings and registers them as a model buffer.

### `GestureTransformer`

Implements:

```text
[B, T, input_dim]
      ↓
Linear(input_dim, hidden_dim)
      ↓
Sinusoidal positional encoding
      ↓
4 × Transformer encoder layers
      ↓
Masked mean pooling
      ↓
Linear(hidden_dim, num_classes)
      ↓
[B, num_classes]
```

It validates padding-mask dtype and shape, rejects all-padded samples, clears
padded encoder outputs with `masked_fill`, and treats remaining NaN/Inf values
as numerical errors.

### `Trainer`

Implements:

- training epochs
- validation epochs
- sample-weighted loss
- classification accuracy
- device movement
- scheduler stepping
- learning-rate reporting
- per-epoch history collection
- best-model checkpoint saving

---

## More information

- `src/gesture_transformer/training/gesture_dataset.py`
- `src/gesture_transformer/training/data_loader.py`
- `src/gesture_transformer/training/gesture_transformer_en.py`
- `src/gesture_transformer/training/transformer_en_trainer.py`
- `04_training_T_encoder.py`
