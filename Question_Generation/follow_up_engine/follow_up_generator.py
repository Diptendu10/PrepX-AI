
class FollowUpGenerator:

    def __init__(self, question_generator):

        self.question_generator = question_generator

    def generate_follow_up(
        self,
        interview_category,
        job_role,
        difficulty_level,
        previous_question,
        previous_answer,
        previous_questions=None,
        previous_answers=None,
        question_history=None,
        evaluation_result=None
    ):

        if not isinstance(previous_question, str) or not previous_question.strip():
            raise ValueError("Previous question must be non-empty text.")

        if not isinstance(previous_answer, str) or not previous_answer.strip():
            raise ValueError("Previous answer must be non-empty text.")

        if previous_questions is None:
            previous_questions = []

        if previous_answers is None:
            previous_answers = []

        if question_history is None:
            question_history = []

        if not isinstance(previous_questions, list):
            raise ValueError("Previous questions must be a list.")

        if not isinstance(previous_answers, list):
            raise ValueError("Previous answers must be a list.")

        if not isinstance(question_history, list):
            raise ValueError("Question history must be a list.")

        # Ensure the latest question is included in history.
        if previous_question not in previous_questions:
            previous_questions = previous_questions + [previous_question]

        # Ensure the latest answer is included in history.
        if previous_answer not in previous_answers:
            previous_answers = previous_answers + [previous_answer]

        # Generate a follow-up using the Question Generator.
        question = self.question_generator.generate_question(
            interview_category=interview_category,
            job_role=job_role,
            difficulty_level=difficulty_level,
            previous_questions=previous_questions,
            previous_answers=previous_answers,
            previous_answer=previous_answer,
            question_history=question_history,
            question_type="follow_up",
            evaluation_result=evaluation_result
        )

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Follow-up generation returned an empty question.")

        return question.strip()
