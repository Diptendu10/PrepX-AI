
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Question_Generation.core.tokenizer import Tokenizer
from Question_Generation.core.language_model import LanguageModel
from Question_Generation.core.generation import TextGenerator


# Step 1: Prepare sample vocabulary
texts = [
    "What is Python?",
    "Explain machine learning.",
    "What is artificial intelligence?",
    "Explain neural networks."
]

tokenizer = Tokenizer()
tokenizer.build_vocab(texts)

# Step 2: Create Language Model
model = LanguageModel(
    vocab_size=len(tokenizer.get_vocab()),
    embedding_dim=16,
    num_heads=4,
    num_layers=2,
    hidden_dim=64,
    max_length=128
)

# Step 3: Create Text Generator
generator = TextGenerator(
    model=model,
    tokenizer=tokenizer
)

# Step 4: Generate text
prompt = "What is"

generated_text = generator.generate(
    prompt=prompt,
    max_new_tokens=10,
    temperature=1.0
)

# Step 5: Display results
print("\n========== PREPX AI ==========")

print("\nStarting Prompt:")
print(prompt)

print("\nGenerated Text:")
print(generated_text)

print("\nText Generation Engine Working Successfully!")
