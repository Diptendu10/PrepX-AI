
import torch
import torch.nn as nn

from Question_Generation.core.generation import TextGenerator


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.max_length = 16
        self.dummy_parameter = nn.Parameter(torch.zeros(1))

    def forward(self, input_tensor):
        batch_size, sequence_length = input_tensor.shape

        logits = torch.zeros(
            batch_size,
            sequence_length,
            10,
            device=input_tensor.device
        )

        # Make token ID 4 the most likely next token.
        logits[:, :, 4] = 10.0

        return logits


class MockTokenizer:
    def __init__(self):
        self.vocab = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "generated": 4
        }

    def encode(self, text):
        return [2, 5]

    def decode(self, token_ids):
        if not token_ids:
            return ""

        return "generated"


def main():
    model = MockModel()
    tokenizer = MockTokenizer()

    generator = TextGenerator(model, tokenizer)

    # Test generation.
    result = generator.generate(
        "Sample prompt",
        max_new_tokens=3,
        temperature=0.8,
        top_k=1
    )

    assert result == "generated"

    # Test top-k sampling.
    logits = torch.tensor([[1.0, 2.0, 5.0, 3.0]])

    selected = generator._sample_next_token(
        logits,
        temperature=1.0,
        top_k=1
    )

    assert selected.item() == 2

    # Test invalid temperature.
    try:
        generator.generate("Sample prompt", temperature=0)
        raise AssertionError("Invalid temperature was accepted.")
    except ValueError:
        pass

    # Test invalid top-k.
    try:
        generator.generate("Sample prompt", top_k=0)
        raise AssertionError("Invalid top-k was accepted.")
    except ValueError:
        pass

    print("Advanced Text Generation Working Successfully!")


if __name__ == "__main__":
    main()
