
from Question_Generation.question_engine.interview_question_controller import (
    InterviewQuestionController
)


class MockQuestionGenerator:

    def __init__(self):
        self.call_count = 0

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

        self.call_count += 1

        if self.call_count == 1:
            return "Bad"

        return "What is object-oriented programming?"


def main():

    mock_generator = MockQuestionGenerator()

    controller = InterviewQuestionController(
        mock_generator
    )

    question = controller.start_interview(
        interview_category="Technical",
        job_role="Python Developer",
        difficulty_level="easy"
    )

    assert question == "What is object-oriented programming?"

    assert mock_generator.call_count == 2

    assert controller.context.current_question == question

    print("Automatic Question Regeneration Working Successfully!")


if __name__ == "__main__":
    main()
