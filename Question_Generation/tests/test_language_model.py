
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Question_Generation.core.tokenizer import Tokenizer
from Question_Generation.core.language_model import LanguageModel


# Step 1: Prepare sample text
texts = [
    "What is Python?",
    "Explain machine learning.",
    "What is artificial intelligence?"
]

tokenizer = Tokenizer()
tokenizer.build_vocab(texts)

# Step 2: Encode input
input_text = "What is Python?"

token_ids = tokenizer.encode(input_text)

input_tensor = torch.tensor(
    [token_ids],
    dtype=torch.long
)

# Step 3: Create Language Model
vocab_size = len(tokenizer.get_vocab())

model = LanguageModel(
    vocab_size=vocab_size,
    embedding_dim=16,
    num_heads=4,
    num_layers=2,
    hidden_dim=64,
    max_length=128
)

# Step 4: Forward pass
model.eval()

with torch.no_grad():
    logits = model(input_tensor)

# Step 5: Get predicted token IDs
predicted_tokens = torch.argmax(
    logits,
    dim=-1
)

# Step 6: Display results
print("\n========== PREPX AI ==========")

print("\nInput Text:")
print(input_text)

print("\nInput Tensor Shape:")
print(input_tensor.shape)

print("\nVocabulary Size:")
print(vocab_size)

print("\nNumber of Transformer Blocks:")
print(len(model.transformer_blocks))

print("\nOutput Logits Shape:")
print(logits.shape)

print("\nPredicted Token IDs Shape:")
print(predicted_tokens.shape)

print("\nTotal Model Parameters:")
print(sum(p.numel() for p in model.parameters()))

# Step 7: Validate output
assert logits.shape == (
    1,
    input_tensor.size(1),
    vocab_size
)

assert torch.isfinite(logits).all()

print("\nLanguage Model Working Successfully!")
