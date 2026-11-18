
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Question_Generation.core.causal_mask import create_causal_mask


sequence_length = 6

mask = create_causal_mask(sequence_length)

print("\n========== PREPX AI ==========")

print("\nCausal Mask:")
print(mask)

print("\nMask Shape:")
print(mask.shape)

print("\nCausal Mask Working Successfully!")
