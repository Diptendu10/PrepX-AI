
from Question_Generation.question_engine.interview_question_controller import (
    InterviewQuestionController
)


class MockQuestionGenerator:

    def generate_question(
        self,
        interview_category,
        job_role,
        difficulty_level,
        previous_questions=None,
        previous_answer=None,
        question_type="initial",
        evaluation_result=None
    ):

        if question_type == "initial":
            return "What is Python?"

        if question_type == "follow_up":
            return "Can you explain your previous answer?"

        return "What are Python's main features?"


def main():

    mock_generator = MockQuestionGenerator()

    controller = InterviewQuestionController(
        mock_generator
    )

    first_question = controller.start_interview(
        interview_category="Technical",
        job_role="Python Developer",
        difficulty_level="easy"
    )

    assert first_question == "What is Python?"

    controller.submit_answer(
        "Python is a programming language."
    )

    evaluation_result = {
        "score": 7,
        "feedback": "Good basic explanation."
    }

    follow_up = controller.generate_follow_up(
        evaluation_result
    )

    assert follow_up == "Can you explain your previous answer?"

    controller.submit_answer(
        "Python is used for software development."
    )

    next_question = controller.generate_next_question()

    assert next_question == "What are Python's main features?"

    print("Interview Question Controller Working Successfully!")


if __name__ == "__main__":
    main()

