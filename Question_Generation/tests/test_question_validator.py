
from Question_Generation.validation.question_validator import (
    QuestionValidator
)


def main():

    validator = QuestionValidator()

    # Valid question
    result = validator.validate(
        "What is the difference between Python and Java?"
    )

    assert result["valid"] is True

    # Empty question
    result = validator.validate("")

    assert result["valid"] is False

    # Too short
    result = validator.validate("Why?")

    assert result["valid"] is False

    # Missing question mark
    result = validator.validate(
        "Explain the concept of object oriented programming"
    )

    assert result["valid"] is False

    # Duplicate question
    result = validator.validate(
        "What is the difference between Python and Java?",
        previous_questions=[
            "What is the difference between Python and Java?"
        ]
    )

    assert result["valid"] is False

    # Whitespace-normalized duplicate
    result = validator.validate(
        "What is the difference between Python and Java?",
        previous_questions=[
            "  What is the difference between Python and Java?  "
        ]
    )

    assert result["valid"] is False

    # Invalid output type
    result = validator.validate(None)

    assert result["valid"] is False

    print("Question Validator Working Successfully!")


if __name__ == "__main__":
    main()
