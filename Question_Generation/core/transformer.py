
import torch
import torch.nn as nn

from Question_Generation.core.attention import MultiHeadAttention
from Question_Generation.core.causal_mask import create_causal_mask


class FeedForwardNetwork(nn.Module):

    def __init__(
        self,
        embedding_dim,
        hidden_dim=64
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim,
                embedding_dim
            )

        )

    def forward(self, x):

        return self.network(x)


class ResidualConnection(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, sublayer_output):

        return input_tensor + sublayer_output


class LayerNormalization(nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()

        self.normalization = nn.LayerNorm(
            embedding_dim
        )

    def forward(self, x):

        return self.normalization(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim=64
    ):

        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads
        )

        self.feed_forward = FeedForwardNetwork(
            embedding_dim,
            hidden_dim
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, attention_mask=None):

        # Step 1: Get sequence length
        sequence_length = x.size(1)

        # Step 2: Generate causal mask automatically
        if attention_mask is None:

            attention_mask = create_causal_mask(
                sequence_length
            )

        # Step 3: Prepare mask for multi-head attention
        attention_mask = attention_mask.to(
            device=x.device,
            dtype=torch.bool
        )

        attention_mask = attention_mask[
            None, None, :, :
        ]

        # Step 4: Attention sublayer
        normalized_x = self.norm1(x)

        attention_output, attention_weights = self.attention(
            normalized_x,
            attention_mask
        )

        # Step 5: Residual connection
        x = x + attention_output

        # Step 6: Feed-forward sublayer
        normalized_x = self.norm2(x)

        feed_forward_output = self.feed_forward(
            normalized_x
        )

        # Step 7: Residual connection
        x = x + feed_forward_output

        return x, attention_weights
