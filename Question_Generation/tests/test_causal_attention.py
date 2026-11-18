
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Question_Generation.core.tokenizer import Tokenizer
from Question_Generation.core.embeddings import EmbeddingLayer
from Question_Generation.core.positional_encoding import PositionalEncoding
from Question_Generation.core.transformer import TransformerBlock


# Step 1: Prepare sample text
texts = [
    "What is Python?"
]

tokenizer = Tokenizer()
tokenizer.build_vocab(texts)

token_ids = tokenizer.encode(texts[0])

input_tensor = torch.tensor(
    [token_ids],
    dtype=torch.long
)

# Step 2: Create embeddings
embedding_dim = 16

embedding_layer = EmbeddingLayer(
    len(tokenizer.get_vocab()),
    embedding_dim
)

embeddings = embedding_layer(input_tensor)

# Step 3: Add positional encoding
positional_encoding = PositionalEncoding(
    embedding_dim
)

position_embeddings = positional_encoding(
    embeddings
)

# Step 4: Create Transformer Block
transformer = TransformerBlock(
    embedding_dim=16,
    num_heads=4,
    hidden_dim=64
)

# Step 5: Forward pass
output, attention_weights = transformer(
    position_embeddings
)

# Step 6: Check output shapes
print("\n========== PREPX AI ==========")

print("\nInput Shape:")
print(input_tensor.shape)

print("\nTransformer Output Shape:")
print(output.shape)

print("\nAttention Weights Shape:")
print(attention_weights.shape)

# Step 7: Check future-token attention
sequence_length = input_tensor.size(1)

future_mask = torch.triu(
    torch.ones(sequence_length, sequence_length),
    diagonal=1
).bool()

future_weights = attention_weights.masked_select(
    future_mask.unsqueeze(0).unsqueeze(0)
)

assert torch.all(
    future_weights == 0
), "Future token attention was not blocked!"

print("\nFuture Token Attention:")
print("Successfully Blocked!")

print("\nCausal Attention Working Successfully!")
