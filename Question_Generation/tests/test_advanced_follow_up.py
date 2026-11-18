
from Question_Generation.follow_up_engine.follow_up_generator import (
    FollowUpGenerator
)


class MockQuestionGenerator:

    def __init__(self):
        self.last_arguments = None

    def generate_question(self, **kwargs):

        self.last_arguments = kwargs

        return "Can you explain how you would optimize that solution?"


def test_follow_up_generation():

    mock_question_generator = MockQuestionGenerator()

    follow_up_generator = FollowUpGenerator(
        mock_question_generator
    )

    question = follow_up_generator.generate_follow_up(
        interview_category="Technical",
        job_role="Python Developer",
        difficulty_level="medium",
        previous_question="What is Python?",
        previous_answer="Python is a high-level programming language.",
        previous_questions=["What is Python?"],
        previous_answers=[
            "Python is a high-level programming language."
        ],
        question_history=[
            {
                "question": "What is Python?",
                "answer": "Python is a high-level programming language."
            }
        ],
        evaluation_result={
            "score": 8,
            "feedback": "Good understanding."
        }
    )

    assert isinstance(question, str)
    assert len(question) > 0

    assert mock_question_generator.last_arguments["question_type"] == (
        "follow_up"
    )

    assert mock_question_generator.last_arguments["previous_answer"] == (
        "Python is a high-level programming language."
    )

    assert mock_question_generator.last_arguments["evaluation_result"] == {
        "score": 8,
        "feedback": "Good understanding."
    }

    print("Advanced Follow-Up Test Passed!")


def test_invalid_answer():

    follow_up_generator = FollowUpGenerator(
        MockQuestionGenerator()
    )

    try:
        follow_up_generator.generate_follow_up(
            interview_category="Technical",
            job_role="Python Developer",
            difficulty_level="medium",
            previous_question="What is Python?",
            previous_answer=""
        )

    except ValueError:
        print("Invalid Answer Test Passed!")
        return

    raise AssertionError("Expected ValueError was not raised.")


if __name__ == "__main__":

    test_follow_up_generation()
    test_invalid_answer()

    print("\nAll Advanced Follow-Up Tests Passed!")
