
from Question_Generation.config.model_config import ModelConfig


def main():

    assert ModelConfig.EMBEDDING_DIM == 64

    assert ModelConfig.NUM_HEADS == 4

    assert ModelConfig.NUM_LAYERS == 2

    assert ModelConfig.HIDDEN_DIM == 128

    assert ModelConfig.MAX_NEW_TOKENS == 30

    assert ModelConfig.TEMPERATURE == 0.8

    assert ModelConfig.MAX_REGENERATION_ATTEMPTS == 3

    assert "easy" in ModelConfig.SUPPORTED_DIFFICULTIES

    assert "medium" in ModelConfig.SUPPORTED_DIFFICULTIES

    assert "hard" in ModelConfig.SUPPORTED_DIFFICULTIES

    print("Question Generation Model Configuration Working Successfully!")


if __name__ == "__main__":
    main()
