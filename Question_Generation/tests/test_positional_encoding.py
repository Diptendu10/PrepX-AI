import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Question_Generation.core.tokenizer import Tokenizer
from Question_Generation.core.embeddings import EmbeddingLayer
from Question_Generation.core.positional_encoding import PositionalEncoding


# -----------------------------------
# STEP 1: Sample interview questions
# -----------------------------------

texts = [
    "What is Python?",
    "Explain object oriented programming.",
    "What is inheritance?"
]


# -----------------------------------
# STEP 2: Initialize tokenizer
# -----------------------------------

tokenizer = Tokenizer()

tokenizer.build_vocab(texts)

vocab_size = len(tokenizer.get_vocab())


# -----------------------------------
# STEP 3: Convert sentence into IDs
# -----------------------------------

sentence = "What is Python?"

token_ids = tokenizer.encode(sentence)

input_tensor = torch.tensor(
    [token_ids],
    dtype=torch.long
)


# -----------------------------------
# STEP 4: Create embedding layer
# -----------------------------------

embedding_dim = 16

embedding_layer = EmbeddingLayer(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim
)


# -----------------------------------
# STEP 5: Generate token embeddings
# -----------------------------------

embeddings = embedding_layer(input_tensor)


# -----------------------------------
# STEP 6: Create positional encoding
# -----------------------------------

positional_encoding = PositionalEncoding(
    embedding_dim=embedding_dim,
    max_length=512
)


# -----------------------------------
# STEP 7: Apply positional encoding
# -----------------------------------

final_output = positional_encoding(embeddings)


# -----------------------------------
# STEP 8: Display results
# -----------------------------------

print("\n========== PREPX AI ==========")

print("\nInput Sentence:")
print(sentence)

print("\nToken IDs:")
print(input_tensor)

print("\nEmbedding Shape:")
print(embeddings.shape)

print("\nPositional Encoding Shape:")
print(positional_encoding.positional_encoding.shape)

print("\nFinal Output Shape:")
print(final_output.shape)

print("\nOriginal Embeddings:")
print(embeddings)

print("\nFinal Embeddings with Position:")
print(final_output)

print("\nPositional Encoding Working Successfully!")