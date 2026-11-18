
import tempfile
import torch

from Question_Generation.core.model_manager import ModelManager


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = ModelManager(checkpoint_dir=temp_dir)

        model = manager.initialize_model(vocab_size=50)

        tokenizer_vocab = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[BOS]": 2,
            "[EOS]": 3
        }

        manager.save_checkpoint(model, tokenizer_vocab)

        loaded_model, loaded_vocab = manager.load_model()

        assert loaded_model.vocab_size == 50
        assert loaded_vocab == tokenizer_vocab

        for original, loaded in zip(
            model.parameters(),
            loaded_model.parameters()
        ):
            assert torch.equal(original, loaded)

    print("Model Manager Working Successfully!")


if __name__ == "__main__":
    main()

