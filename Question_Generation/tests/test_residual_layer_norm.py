
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Question_Generation.core.transformer import (
    ResidualConnection,
    LayerNormalization
)


# Configuration
batch_size = 1
sequence_length = 6
embedding_dim = 16

# Create sample input
input_tensor = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)

sublayer_output = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)

# Residual Connection
residual = ResidualConnection()

residual_output = residual(
    input_tensor,
    sublayer_output
)

# Layer Normalization
layer_norm = LayerNormalization(
    embedding_dim
)

final_output = layer_norm(
    residual_output
)

# Results
print("\n========== PREPX AI ==========")

print("\nInput Shape:")
print(input_tensor.shape)

print("\nResidual Output Shape:")
print(residual_output.shape)

print("\nFinal Output Shape:")
print(final_output.shape)

print("\nLayer Normalization Working Successfully!")
