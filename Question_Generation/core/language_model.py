
import torch
import torch.nn as nn

from Question_Generation.core.embeddings import EmbeddingLayer
from Question_Generation.core.positional_encoding import PositionalEncoding
from Question_Generation.core.transformer import TransformerBlock
from Question_Generation.config.model_config import ModelConfig


class LanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size=None,
        embedding_dim=None,
        num_heads=None,
        num_layers=None,
        hidden_dim=None,
        max_length=None
    ):

        super().__init__()

        # Load centralized configuration

        self.vocab_size = (
            vocab_size
            if vocab_size is not None
            else ModelConfig.VOCAB_SIZE
        )

        self.embedding_dim = (
            embedding_dim
            if embedding_dim is not None
            else ModelConfig.EMBEDDING_DIM
        )

        self.num_heads = (
            num_heads
            if num_heads is not None
            else ModelConfig.NUM_HEADS
        )

        self.num_layers = (
            num_layers
            if num_layers is not None
            else ModelConfig.NUM_LAYERS
        )

        self.hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else ModelConfig.HIDDEN_DIM
        )

        self.max_length = (
            max_length
            if max_length is not None
            else ModelConfig.MAX_SEQUENCE_LENGTH
        )

        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                "Embedding dimension must be divisible by number of heads."
            )

        self.token_embedding = EmbeddingLayer(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim
        )

        self.positional_encoding = PositionalEncoding(
            embedding_dim=self.embedding_dim,
            max_length=self.max_length
        )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim
            )
            for _ in range(self.num_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(
            self.embedding_dim
        )

        self.output_layer = nn.Linear(
            self.embedding_dim,
            self.vocab_size
        )

    def forward(self, token_ids):

        if token_ids.size(1) > self.max_length:
            raise ValueError(
                "Input sequence exceeds maximum sequence length."
            )

        x = self.token_embedding(token_ids)

        x = self.positional_encoding(x)

        for transformer_block in self.transformer_blocks:

            x, attention_weights = transformer_block(x)

        x = self.final_layer_norm(x)

        logits = self.output_layer(x)

        return logits
