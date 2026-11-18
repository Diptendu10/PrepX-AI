
import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size, embedding_dim, padding_idx=0):

        super().__init__()

        self.embedding_dim = embedding_dim

        # Create trainable embedding matrix
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

    def forward(self, token_ids):

        # Convert token IDs into embedding vectors
        embedded_vectors = self.embedding(token_ids)

        return embedded_vectors
