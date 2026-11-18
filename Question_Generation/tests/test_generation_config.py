
import torch

from Question_Generation.core.generation import TextGenerator


class MockModel:
    def __init__(self):
        self.max_length = 128
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, input_tensor):
        batch_size, sequence_length = input_tensor.shape

        # Return mock logits for a vocabulary of 10 tokens.
        logits = torch.zeros(batch_size, sequence_length, 10)

        # Give token ID 3 the highest probability.
        logits[:, :, 3] = 10.0

        return logits


class MockTokenizer:
    def __init__(self):
        self.vocab = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3
        }

    def encode(self, text):
        return [2, 4, 5]

    def decode(self, token_ids):
        return "Sample generated question?"


def main():
    model = MockModel()
    tokenizer = MockTokenizer()

    generator = TextGenerator(model, tokenizer)

    assert generator.max_new_tokens > 0
    assert generator.temperature > 0

    result = generator.generate("Sample prompt")

    assert isinstance(result, str)
    assert result == "Sample generated question?"
    assert model.eval_called

    print("Text Generator Configuration Working Successfully!")


if __name__ == "__main__":
    main()
