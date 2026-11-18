import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim, max_length=512):

        super().__init__()

        positional_encoding = torch.zeros(
            max_length, embedding_dim
        )

        positions = torch.arange(
            0, max_length
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2)
            * (-math.log(10000.0) / embedding_dim)
        )

        positional_encoding[:, 0::2] = torch.sin(
            positions * div_term
        )

        positional_encoding[:, 1::2] = torch.cos(
            positions * div_term
        )

        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer(
            "positional_encoding",
            positional_encoding
        )

    def forward(self, embeddings):

        sequence_length = embeddings.size(1)

        position_values = self.positional_encoding[
            :, :sequence_length, :
        ]

        return embeddings + position_values