
import torch

from Question_Generation.core.language_model import LanguageModel
from Question_Generation.config.model_config import ModelConfig


def main():

    model = LanguageModel()

    assert model.vocab_size == ModelConfig.VOCAB_SIZE

    assert model.embedding_dim == ModelConfig.EMBEDDING_DIM

    assert model.num_heads == ModelConfig.NUM_HEADS

    assert model.num_layers == ModelConfig.NUM_LAYERS

    assert model.hidden_dim == ModelConfig.HIDDEN_DIM

    assert model.max_length == ModelConfig.MAX_SEQUENCE_LENGTH

    sample_input = torch.randint(
        low=0,
        high=ModelConfig.VOCAB_SIZE,
        size=(1, 10)
    )

    output = model(sample_input)

    assert output.shape == (
        1,
        10,
        ModelConfig.VOCAB_SIZE
    )

    print("Language Model Configuration Integration Successful!")


if __name__ == "__main__":
    main()
