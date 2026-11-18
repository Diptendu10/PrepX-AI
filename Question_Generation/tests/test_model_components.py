
import torch

from Question_Generation.core.tokenizer import Tokenizer
from Question_Generation.core.language_model import LanguageModel

from Question_Generation.difficulty_control.difficulty_manager import (
    DifficultyManager
)

from Question_Generation.validation.question_validator import (
    QuestionValidator
)


def test_tokenizer():

    tokenizer = Tokenizer()

    tokenizer.build_vocab([
        "What is Python?",
        "Explain machine learning."
    ])

    encoded = tokenizer.encode("What is Python?")

    assert len(encoded) > 0
    assert isinstance(encoded, list)

    print("Tokenizer Unit Test Passed!")


def test_language_model():

    tokenizer = Tokenizer()

    tokenizer.build_vocab([
        "What is Python?",
        "Explain machine learning."
    ])

    model = LanguageModel(
        vocab_size=len(tokenizer.get_vocab()),
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=64,
        max_length=32
    )

    input_ids = torch.tensor([
        tokenizer.encode("What is Python?")
    ])

    model.eval()

    with torch.no_grad():
        output = model(input_ids)

    assert output.shape[0] == 1
    assert output.shape[1] == input_ids.shape[1]
    assert output.shape[2] == len(tokenizer.get_vocab())

    print("Language Model Unit Test Passed!")


def test_difficulty_manager():

    manager = DifficultyManager()

    assert manager.normalize_difficulty("Easy") == "easy"
    assert manager.normalize_difficulty("Medium") == "medium"
    assert manager.normalize_difficulty("Hard") == "hard"

    print("Difficulty Manager Unit Test Passed!")


def test_question_validator():

    validator = QuestionValidator()

    result = validator.validate(
        "What is object oriented programming?"
    )

    assert result["valid"] is True

    print("Question Validator Unit Test Passed!")


if __name__ == "__main__":

    test_tokenizer()

    test_language_model()

    test_difficulty_manager()

    test_question_validator()

    print("\nAll Model Component Tests Passed!")
