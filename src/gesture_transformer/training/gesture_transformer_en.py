import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        max_len: int = 40,  # frames
    ):
        super().__init__()

        # Initialize positional encoding matrix
        position = torch.arange(
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        # Calculate sinusoidal values for each dimension
        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices and cosine to odd indices
        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding to embeddings
        sequence_length = x.shape[1]

        pe = self.get_buffer("pe")

        return x + pe[:, :sequence_length]


class GestureTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        max_len: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_classes = num_classes

        # [B, T, 126] -> [B, T, 64]
        self.projection = nn.Linear(
            input_dim,
            hidden_dim,
        )

        # Adds temporal position information.
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=hidden_dim,
            max_len=max_len,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # [B, 64] -> [B, 5]
        self.classifier = nn.Linear(
            hidden_dim,
            self.num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run gesture classification.

        Args:
            x:
                Input features with shape [B, T, input_dim].

            padding_mask:
                Boolean tensor with shape [B, T].

                False = valid sequence position.
                True = padded sequence position.

                This follows PyTorch src_key_padding_mask
                semantics. Raw masks stored in train.pt use
                the opposite convention and must be converted
                before being passed to the model.

        Returns:
            Raw classification logits with shape
            [B, num_classes].
        """

        # Ensure the model receives the mask format expected by
        # PyTorch's src_key_padding_mask.
        if padding_mask.dtype != torch.bool:
            raise TypeError(
                "padding_mask must be a boolean tensor where True means padding."
            )

        if padding_mask.shape != x.shape[:2]:
            raise ValueError(
                "padding_mask must have shape [B, T]. "
                f"Expected {tuple(x.shape[:2])}, "
                f"got {tuple(padding_mask.shape)}."
            )

        # Every gesture must contain at least one real frame.
        # An all-padded sequence can cause attention to operate
        # without any valid keys and may produce NaN values.
        if padding_mask.all(dim=1).any():
            raise ValueError("Every gesture must contain at least one valid frame.")

        # [B, T, input_dim] -> [B, T, hidden_dim]
        x = self.projection(x)

        # Add temporal position information.
        x = self.positional_encoding(x)

        # [B, T, hidden_dim] -> [B, T, hidden_dim]
        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        # Explicitly overwrite padded encoder outputs.
        #
        # Do not use:
        #     x = x * valid_mask
        #
        # because NaN * 0 is still NaN.
        x = x.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )

        # At this point padded positions have been removed.
        # Any remaining NaN/Inf belongs to a valid frame and
        # indicates a real numerical problem.
        if not torch.isfinite(x).all():
            raise RuntimeError(
                "Transformer encoder produced NaN or Inf for one or more valid frames."
            )

        # Count the number of real frames in each gesture.
        valid_frame_count = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)

        # Masked mean pooling:
        # [B, T, hidden_dim] -> [B, hidden_dim]
        x = x.sum(dim=1) / valid_frame_count

        # [B, hidden_dim] -> [B, num_classes]
        logits = self.classifier(x)

        return logits
