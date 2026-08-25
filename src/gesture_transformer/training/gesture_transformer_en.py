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
        input_dim: int = 126,
        hidden_dim: int = 64,
        max_len: int = 40,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        num_classes: int = 5,
    ):
        super().__init__()

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
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:

        # [B, T, 126] -> [B, T, 64]
        x = self.projection(x)

        # [B, T, 64] -> [B, T, 64]
        x = self.positional_encoding(x)

        # [B, T, 64] -> [B, T, 64]
        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        # padding_mask:
        # False = valid
        # True  = padding
        #
        # valid_mask:
        # True  = valid
        # False = padding
        valid_mask = (~padding_mask).unsqueeze(-1)

        # Explicitly convert mask to the same dtype as x.
        valid_mask = valid_mask.to(x.dtype)

        # Remove padded positions from pooling.
        x = x * valid_mask

        # Number of valid frames in every gesture.
        valid_frame_count = valid_mask.sum(dim=1).clamp(min=1.0)

        # [B, T, 64] -> [B, 64]
        x = x.sum(dim=1) / valid_frame_count

        # [B, 64] -> [B, 5]
        logits = self.classifier(x)

        return logits
