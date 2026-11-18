
import sys
from pathlib import Path

import torch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Question_Generation.core.tokenizer import Tokenizer
from Question_Generation.core.embeddings import EmbeddingLayer


# Temporary testing sentences
texts = [
    "What is Python?",
    "Explain object oriented programming.",
    "What is inheritance?"
]

# Create tokenizer
tokenizer = Tokenizer()
tokenizer.build_vocab(texts)

# Get vocabulary size
vocab_size = len(tokenizer.get_vocab())

# Convert sentence into token IDs
sentence = "What is Python?"
token_ids = tokenizer.encode(sentence)

# Convert IDs into PyTorch tensor
input_tensor = torch.tensor([token_ids], dtype=torch.long)

# Create embedding layer
embedding_dim = 16

embedding_layer = EmbeddingLayer(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim
)

# Generate embeddings
output = embedding_layer(input_tensor)

print("Vocabulary Size:", vocab_size)
print("Embedding Dimension:", embedding_dim)

print("\nInput Token IDs:")
print(input_tensor)

print("\nEmbedding Output Shape:")
print(output.shape)

print("\nEmbedding Vectors:")
print(output)

print("\nEmbedding Layer Working Successfully!")
