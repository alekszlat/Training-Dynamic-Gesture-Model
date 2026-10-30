from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def save_confusion_matrices(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    label_to_index: dict[str, int],
    output_dir: Path,
) -> None:
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in data_loader:
        features = batch["features"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        logits = model(features, padding_mask)
        predictions = logits.argmax(dim=1)

        # Collect one actual/predicted label per gesture sequence.
        y_true.extend(batch["answers"].cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())

    if not y_true:
        raise ValueError("Cannot evaluate an empty data loader.")

    # Match displayed names to the model's numeric class indices.
    class_names = [
        label
        for label, index in sorted(
            label_to_index.items(),
            key=lambda item: item[1],
        )
    ]
    class_ids = list(range(len(class_names)))

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, normalize, values_format in [
        ("counts", None, "d"),
        ("normalized", "true", ".2f"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 7))

        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            labels=class_ids,
            display_labels=class_names,
            normalize=normalize,
            values_format=values_format,
            cmap="Blues",
            xticks_rotation=45,
            ax=ax,
        )

        ax.set_title(f"Validation confusion matrix ({name})")
        fig.tight_layout()

        fig.savefig(
            output_dir / f"confusion_matrix_{name}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)