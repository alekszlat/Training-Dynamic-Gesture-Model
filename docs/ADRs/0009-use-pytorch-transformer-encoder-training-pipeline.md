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

Gesture sequences are normalized to a maximum length of 40 frames.

The feature tensor therefore follows:

`[N, 40, 126]`

where:

- `N` = number of samples
- `40` = sequence length
- `126` = features per frame

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

The training entry point derives `num_classes` dynamically from
`label_to_index.json`; the classifier output size follows that value.

Shorter sequences contain padded positions. These positions must not influence Transformer attention or final sequence pooling.

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
[N, 40, 126]

answers:
[N]

padding_mask:
[N, 40]
```

A single dataset sample contains:

```text
features:
[40, 126]

answers:
scalar class index

padding_mask:
[40]

sample_id:
string

label:
string
```

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

The stored mask is used instead of deriving padding from zero-valued feature vectors because a valid frame may still contain an all-zero feature vector.

---

## DataLoader

DataLoaders are created using `create_data_loader`.

Current configuration:

```text
batch_size = 32
num_workers = 0
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
[B, 40, 126]

answers:
[B]

padding_mask:
[B, 40]
```

The DataLoader also batches `sample_id` and `label` metadata.

---

## Device handling

The training script selects the device with:

```python
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

The model is moved with:

```python
model = GestureTransformer().to(DEVICE)
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

## Model architecture

The current `GestureTransformer` defaults are:

```text
input_dim        = 126
hidden_dim       = 64
max_len          = 40
num_layers       = 4
num_heads        = 8
dim_feedforward  = 256
dropout          = 0.2
num_classes      = len(label_to_index)
```

The model pipeline is:

```text
[B, 40, 126]
      ↓
Linear(126, 64)
      ↓
[B, 40, 64]
      ↓
Sinusoidal positional encoding
      ↓
[B, 40, 64]
      ↓
4 × Transformer encoder layers
      ↓
[B, 40, 64]
      ↓
Masked mean pooling
      ↓
[B, 64]
      ↓
Linear(hidden_dim, num_classes)
      ↓
[B, num_classes] logits
```

---

## Linear projection

The model projects each frame from 126 features to 64 features:

```python
self.projection = nn.Linear(
    input_dim,
    hidden_dim,
)
```

This changes:

```text
[B, 40, 126]
```

to:

```text
[B, 40, 64]
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
[B, 40, 64]
→
[B, 40, 64]
```

---

## Masked mean pooling

The Transformer returns:

```text
[B, 40, 64]
```

The padding mask is inverted:

```python
valid_mask = (~padding_mask).unsqueeze(-1)
```

and converted to the same dtype as the encoder output:

```python
valid_mask = valid_mask.to(x.dtype)
```

Padded positions are removed from the pooling sum:

```python
x = x * valid_mask
```

The number of valid frames is calculated with:

```python
valid_frame_count = valid_mask.sum(dim=1).clamp(min=1.0)
```

The final sequence representation is:

```python
x = x.sum(dim=1) / valid_frame_count
```

This reduces:

```text
[B, 40, 64]
```

to:

```text
[B, 64]
```

Only valid frames contribute to the mean.

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

The `Trainer` class receives:

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
model.train()
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
model.eval()
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

## Training duration

The training script currently defines:

```python
NUM_EPOCHS = 15
```

The current invocation still uses:

```python
trainer.train(num_epochs=15)
```

This should eventually be changed to:

```python
trainer.train(num_epochs=NUM_EPOCHS)
```

so the constant becomes the single source of truth.

---

## Best-model checkpointing

Checkpoints are saved under:

```text
src/gesture_transformer/models/checkpoints/
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

| Parameter              | Value                                        |
| ---------------------- | -------------------------------------------- |
| Processed directory    | `data/processed`                             |
| Training file          | `train.pt`                                   |
| Validation file        | `val.pt`                                     |
| Checkpoint directory   | `src/gesture_transformer/models/checkpoints` |
| Checkpoint filename    | `best_model.pth`                             |
| Input dimension        | `126`                                        |
| Sequence length        | `40`                                         |
| Hidden dimension       | `64`                                         |
| Transformer layers     | `4`                                          |
| Attention heads        | `8`                                          |
| Feed-forward dimension | `256`                                        |
| Dropout                | `0.2`                                        |
| Number of classes      | Derived from `label_to_index.json`          |
| Batch size             | `32`                                         |
| DataLoader workers     | `0`                                          |
| Epochs                 | `15`                                         |
| Loss                   | `CrossEntropyLoss`                           |
| Optimizer              | `AdamW`                                      |
| Initial learning rate  | `0.0005`                                     |
| Scheduler              | `ReduceLROnPlateau`                          |
| Scheduler mode         | `min`                                        |
| Scheduler factor       | `0.5`                                        |
| Scheduler patience     | `2`                                          |
| Minimum learning rate  | `0.000001`                                   |
| Device                 | CUDA when available, otherwise CPU           |

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

### Negative

- The pipeline is coupled to the `.pt` keys `x`, `y`, `mask`, `sample_ids`, and `labels`.
- `GestureDataset` does not currently enforce feature and answer dtypes.
- Padding-mask validation currently checks sample count but not the complete `[N, T]` shape.
- `GestureDataset.__getitem__` is annotated as `dict[str, torch.Tensor]` even though it also returns strings.
- Batch size is currently hardcoded in the training entry point.
- `NUM_EPOCHS` exists but the literal `15` is still passed to `Trainer.train`.
- Hyperparameters are module-level constants rather than a dedicated configuration object.
- Only the best validation-loss checkpoint is kept.
- Checkpoint restoration is not yet implemented.
- Resumed training is not yet implemented.
- Training history is printed but not stored.
- Evaluation currently provides only loss and accuracy.
- Confusion matrix is not yet implemented.
- Precision, recall, and F1 are not yet implemented.
- Untouched test-set evaluation is not yet implemented.
- Import paths currently mix `gesture_transformer...` and `src.gesture_transformer...`.

---

## Follow-on work

- Use `NUM_EPOCHS` when calling `Trainer.train`.
- Move batch size into a named configuration value.
- Normalize package import paths.
- Validate the complete padding-mask shape against `features.shape[:2]`.
- Correct the `GestureDataset.__getitem__` return annotation.
- Optionally enforce feature and answer dtypes.
- Record per-epoch training history.
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
- Add automated tests for dataset loading, mask conversion, model output, pooling, training, validation, scheduler stepping, device handling, and checkpoint saving.

---

## Confirmation

The current implementation includes:

### `GestureDatasetLoader`

Loads processed `.pt` dictionaries from CPU storage.

### `GestureDataset`

Validates dataset structure, converts the stored mask into PyTorch padding-mask semantics, and exposes individual gesture samples.

### `create_data_loader`

Creates training and validation DataLoaders.

### `SinusoidalPositionalEncoding`

Creates fixed temporal positional encodings and registers them as a model buffer.

### `GestureTransformer`

Implements:

```text
[B, 40, 126]
      ↓
Linear(126, 64)
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

### `Trainer`

Implements:

- training epochs
- validation epochs
- sample-weighted loss
- classification accuracy
- device movement
- scheduler stepping
- learning-rate reporting
- best-model checkpoint saving

---

## More information

- `src/gesture_transformer/training/gesture_dataset.py`
- `src/gesture_transformer/training/data_loader.py`
- `src/gesture_transformer/training/gesture_transformer_en.py`
- `src/gesture_transformer/training/transformer_en_trainer.py`
- `04_training_T_encoder.py`
