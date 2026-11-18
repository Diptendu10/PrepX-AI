
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Query transformation
        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # Key transformation
        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # Value transformation
        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # Output transformation
        self.output = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, embeddings, attention_mask=None):

        # Step 1: Generate Q, K and V
        Q = self.query(embeddings)
        K = self.key(embeddings)
        V = self.value(embeddings)

        # Step 2: Calculate attention scores
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        # Step 3: Scale the scores
        scores = scores / math.sqrt(self.embedding_dim)

        # Step 4: Apply optional attention mask
        if attention_mask is not None:
            scores = scores.masked_fill(
                attention_mask == 0,
                float("-inf")
            )

        # Step 5: Apply Softmax
        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # Step 6: Combine Value vectors
        context = torch.matmul(
            attention_weights,
            V
        )

        # Step 7: Final output transformation
        output = self.output(context)

        return output, attention_weights


class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads):

        super().__init__()

        assert embedding_dim % num_heads == 0, (
            "Embedding dimension must be divisible by number of heads"
        )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Query, Key and Value transformations
        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        # Output transformation
        self.output = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, embeddings, attention_mask=None):

        batch_size, sequence_length, _ = embeddings.shape

        # Step 1: Generate Q, K and V
        Q = self.query(embeddings)
        K = self.key(embeddings)
        V = self.value(embeddings)

        # Step 2: Split into multiple heads
        Q = Q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # Step 3: Calculate scaled attention scores
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Step 4: Apply optional attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.to(
                device=scores.device
            )

            scores = scores.masked_fill(
                attention_mask == 0,
                float("-inf")
            )

        # Step 5: Calculate attention weights
        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # Step 6: Combine Value vectors
        context = torch.matmul(
            attention_weights,
            V
        )

        # Step 7: Combine all attention heads
        context = context.transpose(
            1, 2
        ).contiguous()

        context = context.view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        # Step 8: Final output transformation
        output = self.output(context)

        return output, attention_weights
