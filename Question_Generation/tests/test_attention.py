
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Question_Generation.core.tokenizer import Tokenizer
from Question_Generation.core.embeddings import EmbeddingLayer
from Question_Generation.core.positional_encoding import PositionalEncoding
from Question_Generation.core.attention import SelfAttention


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
# STEP 4: Embedding Layer
# -----------------------------------

embedding_dim = 16

embedding_layer = EmbeddingLayer(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim
)

embeddings = embedding_layer(input_tensor)


# -----------------------------------
# STEP 5: Positional Encoding
# -----------------------------------

positional_encoding = PositionalEncoding(
    embedding_dim=embedding_dim,
    max_length=512
)

position_embeddings = positional_encoding(embeddings)


# -----------------------------------
# STEP 6: Self-Attention
# -----------------------------------

attention_layer = SelfAttention(
    embedding_dim=embedding_dim
)

attention_output, attention_weights = attention_layer(
    position_embeddings
)


# -----------------------------------
# STEP 7: Display results
# -----------------------------------

print("\n========== PREPX AI ==========")

print("\nInput Sentence:")
print(sentence)

print("\nInput Tensor Shape:")
print(input_tensor.shape)

print("\nPosition Embedding Shape:")
print(position_embeddings.shape)

print("\nAttention Output Shape:")
print(attention_output.shape)

print("\nAttention Weights Shape:")
print(attention_weights.shape)

print("\nAttention Weights:")
print(attention_weights)

print("\nSum of Attention Weights for Each Token:")
print(attention_weights.sum(dim=-1))

print("\nSelf-Attention Working Successfully!")
