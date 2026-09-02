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

        # [B, hidden_dim] -> [B, num_classes]
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
                semantics.

                Raw masks stored in the processed .pt files use
                the opposite convention:

                    1 = valid frame
                    0 = padded frame

                They must therefore be converted before being
                passed to the model.

        Returns:
            Raw classification logits with shape
            [B, num_classes].
        """

        # PyTorch Transformer padding masks must be Boolean:
        #
        # False -> valid position
        # True  -> padded position
        if padding_mask.dtype != torch.bool:
            raise TypeError(
                "padding_mask must be a boolean tensor where True means padding."
            )

        # The padding mask must describe exactly one mask value
        # for every sequence position in x.
        #
        # x:
        #     [B, T, input_dim]
        #
        # padding_mask:
        #     [B, T]
        if padding_mask.shape != x.shape[:2]:
            raise ValueError(
                "padding_mask must have shape [B, T]. "
                f"Expected {tuple(x.shape[:2])}, "
                f"got {tuple(padding_mask.shape)}."
            )

        # Every gesture must contain at least one valid frame.
        #
        # An entirely padded sequence has no meaningful keys for
        # attention and cannot produce a meaningful gesture
        # representation.
        if padding_mask.all(dim=1).any():
            raise ValueError("Every gesture must contain at least one valid frame.")

        # ---------------------------------------------------------
        # 1. Feature projection
        # ---------------------------------------------------------
        #
        # [B, T, input_dim]
        #          ↓
        # [B, T, hidden_dim]
        x = self.projection(x)

        # ---------------------------------------------------------
        # 2. Positional encoding
        # ---------------------------------------------------------
        #
        # Transformers do not inherently know the temporal order
        # of the frames, so positional information is added.
        x = self.positional_encoding(x)

        # ---------------------------------------------------------
        # 3. Transformer encoder
        # ---------------------------------------------------------
        #
        # src_key_padding_mask prevents valid sequence positions
        # from attending to padded key positions.
        #
        # Input:
        #     [B, T, hidden_dim]
        #
        # Output:
        #     [B, T, hidden_dim]
        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        # ---------------------------------------------------------
        # 4. Remove padded encoder outputs
        # ---------------------------------------------------------
        #
        # src_key_padding_mask prevents attention TO padded keys,
        # but padded query positions can still receive encoder
        # outputs.
        #
        # Those positions must therefore be explicitly removed
        # before sequence pooling.
        #
        # masked_fill is intentionally used instead of:
        #
        #     x = x * valid_mask
        #
        # because:
        #
        #     NaN * 0 = NaN
        #
        # masked_fill actually overwrites the padded values.
        x = x.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )

        # ---------------------------------------------------------
        # 5. Numerical validation
        # ---------------------------------------------------------
        #
        # Padded positions have already been replaced with zero.
        #
        # Therefore, any remaining NaN or Inf value belongs to a
        # valid sequence position and represents a genuine
        # numerical problem.
        if not torch.isfinite(x).all():
            raise RuntimeError(
                "Transformer encoder produced NaN or Inf for one or more valid frames."
            )

        # ---------------------------------------------------------
        # 6. Count valid frames
        # ---------------------------------------------------------
        #
        # padding_mask:
        #
        #     False False False True True
        #
        # ~padding_mask:
        #
        #     True True True False False
        #
        # Sum:
        #
        #     3 valid frames
        #
        # Shape:
        #     [B, 1]
        valid_frame_count = (~padding_mask).sum(dim=1, keepdim=True).to(x.dtype)

        # ---------------------------------------------------------
        # 7. Masked mean pooling
        # ---------------------------------------------------------
        #
        # Padded positions are already zero, so summing across the
        # temporal dimension includes only valid frame values.
        #
        # [B, T, hidden_dim]
        #          ↓
        # [B, hidden_dim]
        x = x.sum(dim=1) / valid_frame_count

        # ---------------------------------------------------------
        # 8. Classification
        # ---------------------------------------------------------
        #
        # [B, hidden_dim]
        #          ↓
        # [B, num_classes]
        logits = self.classifier(x)

        # Return raw logits.
        #
        # Do NOT apply softmax here because CrossEntropyLoss
        # expects raw logits during training.
        return logits
