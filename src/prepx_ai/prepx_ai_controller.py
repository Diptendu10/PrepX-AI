
from src.prepx_ai.question_generation_service import (
    QuestionGenerationService
)


class PrepXAIController:

    def __init__(self, question_service=None):

        # Allow a test service to be injected.
        self.question_service = (
            question_service
            if question_service is not None
            else QuestionGenerationService()
        )

    def start_interview(
        self,
        interview_category,
        job_role,
        difficulty_level="medium"
    ):

        return self.question_service.start_interview(
            interview_category=interview_category,
            job_role=job_role,
            difficulty_level=difficulty_level
        )

    def submit_candidate_answer(self, session_id, answer):

        return self.question_service.submit_answer(
            session_id=session_id,
            answer=answer
        )

    def generate_follow_up(self, session_id):

        return self.question_service.generate_follow_up(
            session_id=session_id
        )

    def generate_next_question(self, session_id):

        return self.question_service.generate_next_question(
            session_id=session_id
        )

    def get_interview_context(self, session_id):

        return self.question_service.get_context(
            session_id=session_id
        )

    def end_interview(self, session_id):

        return self.question_service.reset_interview(
            session_id=session_id
        )
