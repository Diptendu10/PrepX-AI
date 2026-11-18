
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Question_Generation.core.tokenizer import Tokenizer
from Question_Generation.core.embeddings import EmbeddingLayer
from Question_Generation.core.positional_encoding import PositionalEncoding
from Question_Generation.core.attention import MultiHeadAttention
from Question_Generation.core.transformer import FeedForwardNetwork


# Sample interview questions
texts = [
    "What is Python?",
    "Explain object oriented programming.",
    "What is inheritance?"
]

# Tokenizer
tokenizer = Tokenizer()
tokenizer.build_vocab(texts)

vocab_size = len(tokenizer.get_vocab())

# Input sentence
sentence = "What is Python?"

token_ids = tokenizer.encode(sentence)

input_tensor = torch.tensor(
    [token_ids],
    dtype=torch.long
)

# Embedding
embedding_dim = 16

embedding_layer = EmbeddingLayer(
    vocab_size,
    embedding_dim
)

embeddings = embedding_layer(input_tensor)

# Positional Encoding
position_layer = PositionalEncoding(
    embedding_dim
)

position_embeddings = position_layer(embeddings)

# Multi-Head Attention
attention_layer = MultiHeadAttention(
    embedding_dim,
    num_heads=4
)

attention_output, _ = attention_layer(
    position_embeddings
)

# Feed-Forward Network
feed_forward = FeedForwardNetwork(
    embedding_dim=embedding_dim,
    hidden_dim=64
)

output = feed_forward(attention_output)

# Results
print("\n========== PREPX AI ==========")

print("\nAttention Output Shape:")
print(attention_output.shape)

print("\nFeed-Forward Output Shape:")
print(output.shape)

print("\nFeed-Forward Network Working Successfully!")

