
from Question_Generation.core.tokenizer import Tokenizer
from Question_Generation.core.language_model import LanguageModel
from Question_Generation.core.generation import TextGenerator

from Question_Generation.question_engine.question_generator import (
    QuestionGenerator
)

from Question_Generation.question_engine.interview_context import (
    InterviewContext
)

from Question_Generation.follow_up_engine.follow_up_generator import (
    FollowUpGenerator
)

from Question_Generation.validation.question_validator import (
    QuestionValidator
)


class QuestionGenerationService:

    def __init__(self):

        self.sessions = {}

        # Sample vocabulary for initial software integration.
        # Replace this with your trained tokenizer vocabulary later.
        sample_texts = [
            "What is Python?",
            "Explain object oriented programming?",
            "What is machine learning?",
            "Explain data structures?",
            "What is a database?",
            "What are Python decorators?",
            "Explain inheritance?",
            "What is an API?",
            "Explain time complexity?",
            "What is a neural network?"
        ]

        self.tokenizer = Tokenizer()
        self.tokenizer.build_vocab(sample_texts)

        self.model = LanguageModel(
            vocab_size=len(self.tokenizer.get_vocab()),
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            hidden_dim=64,
            max_length=128
        )

        self.model.eval()

        self.text_generator = TextGenerator(
            self.model,
            self.tokenizer
        )

        self.question_generator = QuestionGenerator(
            self.text_generator
        )

        self.follow_up_generator = FollowUpGenerator(
            self.question_generator
        )

        self.validator = QuestionValidator()

    def _get_session(self, session_id):

        if session_id not in self.sessions:
            raise ValueError("Interview session not found.")

        return self.sessions[session_id]

    def _generate_validated_question(self, **kwargs):

        for attempt in range(3):

            question = self.question_generator.generate_question(
                **kwargs
            )

            result = self.validator.validate(question)

            if result["valid"]:
                return result["question"]

        raise RuntimeError(
            "Could not generate a valid question. "
            "The model may need training."
        )

    def start_interview(
        self,
        interview_category,
        job_role,
        difficulty_level="medium"
    ):

        import uuid

        session_id = str(uuid.uuid4())

        context = InterviewContext(
            interview_category=interview_category,
            job_role=job_role,
            difficulty_level=difficulty_level
        )

        question = self._generate_validated_question(
            interview_category=interview_category,
            job_role=job_role,
            difficulty_level=difficulty_level,
            question_type="initial"
        )

        context.add_question(question)

        self.sessions[session_id] = context

        return {
            "session_id": session_id,
            "question": question
        }

    def submit_answer(self, session_id, answer):

        context = self._get_session(session_id)

        context.add_answer(answer)

        return {
            "message": "Answer submitted successfully."
        }

    def generate_follow_up(self, session_id):

        context = self._get_session(session_id)

        question = context.get_latest_question()
        answer = context.get_latest_answer()

        if not question or not answer:
            raise ValueError(
                "A question and answer are required."
            )

        follow_up = self.follow_up_generator.generate_follow_up(
            interview_category=context.interview_category,
            job_role=context.job_role,
            difficulty_level=context.difficulty_level,
            previous_question=question,
            previous_answer=answer,
            question_history=context.question_history
        )

        result = self.validator.validate(follow_up)

        if not result["valid"]:
            raise RuntimeError(
                "Generated follow-up question failed validation."
            )

        context.add_question(result["question"])

        return {
            "question": result["question"]
        }

    def generate_next_question(self, session_id):

        context = self._get_session(session_id)

        if context.current_question is not None:
            raise ValueError(
                "Submit the current answer before requesting "
                "the next question."
            )

        question = self._generate_validated_question(
            interview_category=context.interview_category,
            job_role=context.job_role,
            difficulty_level=context.difficulty_level,
            question_history=context.question_history,
            question_type="next"
        )

        context.add_question(question)

        return {
            "question": question
        }

    def get_context(self, session_id):

        context = self._get_session(session_id)

        return context.get_context()

    def reset_interview(self, session_id):

        context = self._get_session(session_id)

        context.reset()

        return {
            "message": "Interview reset successfully."
        }
