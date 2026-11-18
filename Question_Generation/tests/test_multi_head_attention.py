
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Question_Generation.core.tokenizer import Tokenizer
from Question_Generation.core.embeddings import EmbeddingLayer
from Question_Generation.core.positional_encoding import PositionalEncoding
from Question_Generation.core.attention import MultiHeadAttention


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
    vocab_size, embedding_dim
)

embeddings = embedding_layer(input_tensor)

# Positional Encoding
position_layer = PositionalEncoding(
    embedding_dim
)

position_embeddings = position_layer(embeddings)

# Multi-Head Attention
num_heads = 4

attention_layer = MultiHeadAttention(
    embedding_dim,
    num_heads
)

output, weights = attention_layer(
    position_embeddings
)

# Results
print("\n========== PREPX AI ==========")

print("\nInput Shape:")
print(position_embeddings.shape)

print("\nNumber of Heads:")
print(num_heads)

print("\nDimension Per Head:")
print(embedding_dim // num_heads)

print("\nAttention Output Shape:")
print(output.shape)

print("\nAttention Weights Shape:")
print(weights.shape)

print("\nAttention Weights:")
print(weights)

print("\nAttention Weight Sums:")
print(weights.sum(dim=-1))

print("\nMulti-Head Attention Working Successfully!")
