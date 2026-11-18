
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Question_Generation.core.tokenizer import Tokenizer


# Create tokenizer
tokenizer = Tokenizer()

# Temporary testing sentences (not a training dataset)
texts = [
    "What is Python?",
    "Explain object oriented programming.",
    "What is inheritance?"
]

# Build vocabulary for testing
tokenizer.build_vocab(texts)

# Test sentence
sentence = "What is Python?"

# Tokenize
print("Tokens:")
print(tokenizer.tokenize(sentence))

# Encode
encoded = tokenizer.encode(sentence)

print("\nEncoded IDs:")
print(encoded)

# Decode
decoded = tokenizer.decode(encoded)

print("\nDecoded Text:")
print(decoded)

# Display vocabulary
print("\nVocabulary:")
print(tokenizer.get_vocab())
