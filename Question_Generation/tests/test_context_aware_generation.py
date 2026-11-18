
from Question_Generation.question_engine.interview_context import (
    InterviewContext
)

from Question_Generation.question_engine.question_generator import (
    QuestionGenerator
)


class MockTextGenerator:

    def __init__(self):
        self.last_prompt = None

    def generate(
        self,
        prompt,
        max_new_tokens=None,
        temperature=None
    ):

        self.last_prompt = prompt

        return "How would you improve the performance of your Python application?"


def test_interview_context():

    context = InterviewContext(
        interview_category="Technical",
        job_role="Python Developer",
        difficulty_level="medium"
    )

    context.add_question("What is Python?")

    context.add_answer(
        "Python is a high-level programming language."
    )

    context_data = context.get_context()

    assert context_data["question_count"] == 1

    assert len(context_data["previous_questions"]) == 1

    assert len(context_data["previous_answers"]) == 1

    assert context_data["previous_questions"][0] == "What is Python?"

    assert context_data["previous_answers"][0] == (
        "Python is a high-level programming language."
    )

    print("Interview Context Test Passed!")


def test_context_aware_generation():

    mock_generator = MockTextGenerator()

    question_generator = QuestionGenerator(
        text_generator=mock_generator
    )

    context = InterviewContext(
        interview_category="Technical",
        job_role="Python Developer",
        difficulty_level="medium"
    )

    context.add_question("What is Python?")

    context.add_answer(
        "Python is a high-level programming language."
    )

    context_data = context.get_context()

    question = question_generator.generate_question(
        interview_category=context_data["interview_category"],
        job_role=context_data["job_role"],
        difficulty_level=context_data["difficulty_level"],
        previous_questions=context_data["previous_questions"],
        previous_answers=context_data["previous_answers"],
        previous_answer=context_data["previous_answers"][-1],
        question_history=context_data["question_history"],
        question_type="follow_up"
    )

    assert isinstance(question, str)

    assert len(question) > 0

    assert "Python is a high-level programming language." in (
        mock_generator.last_prompt
    )

    assert "What is Python?" in mock_generator.last_prompt

    assert "follow_up" in mock_generator.last_prompt

    print("Context-Aware Generation Test Passed!")


def test_context_reset():

    context = InterviewContext()

    context.add_question("What is Python?")
    context.add_answer("Python is a programming language.")

    context.reset()

    context_data = context.get_context()

    assert context_data["question_count"] == 0
    assert context_data["question_history"] == []
    assert context_data["current_question"] is None

    print("Interview Context Reset Test Passed!")


if __name__ == "__main__":

    test_interview_context()

    test_context_aware_generation()

    test_context_reset()

    print("\nAll Context-Aware Tests Passed!")
