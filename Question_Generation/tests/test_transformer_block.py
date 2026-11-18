
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Question_Generation.core.tokenizer import Tokenizer
from Question_Generation.core.embeddings import EmbeddingLayer
from Question_Generation.core.positional_encoding import PositionalEncoding
from Question_Generation.core.transformer import TransformerBlock


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

# Input
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

# Transformer Block
transformer = TransformerBlock(
    embedding_dim=16,
    num_heads=4,
    hidden_dim=64
)

output, attention_weights = transformer(
    position_embeddings
)

# Results
print("\n========== PREPX AI ==========")

print("\nInput Shape:")
print(position_embeddings.shape)

print("\nTransformer Output Shape:")
print(output.shape)

print("\nAttention Weights Shape:")
print(attention_weights.shape)

print("\nTransformer Block Working Successfully!")
