
from pathlib import Path

import torch

from Question_Generation.core.language_model import LanguageModel


class ModelManager:
    def __init__(self, checkpoint_dir="Question_Generation/checkpoints", device="cpu"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

    def initialize_model(self, vocab_size):
        model = LanguageModel(vocab_size=vocab_size)
        return model.to(self.device)

    def save_checkpoint(self, model, tokenizer_vocab, filename="question_generation.pt"):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "vocab_size": model.vocab_size,
                "embedding_dim": model.embedding_dim,
                "num_heads": model.num_heads,
                "num_layers": model.num_layers,
                "hidden_dim": model.hidden_dim,
                "max_length": model.max_length
            },
            "tokenizer_vocab": tokenizer_vocab
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        return path

    def load_model(self, filename="question_generation.pt"):
        path = self.checkpoint_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True
        )

        config = checkpoint["model_config"]

        model = LanguageModel(
            vocab_size=config["vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            hidden_dim=config["hidden_dim"],
            max_length=config["max_length"]
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        return model, checkpoint["tokenizer_vocab"]
