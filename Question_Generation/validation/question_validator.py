
import re

from Question_Generation.config.model_config import ModelConfig


class QuestionValidator:

    def __init__(
        self,
        min_length=None,
        max_length=None
    ):

        self.min_length = (
            min_length
            if min_length is not None
            else ModelConfig.MIN_QUESTION_LENGTH
        )

        self.max_length = (
            max_length
            if max_length is not None
            else ModelConfig.MAX_QUESTION_LENGTH
        )

    def validate(
        self,
        question,
        previous_questions=None,
        interview_category=None,
        job_role=None
    ):

        if not isinstance(question, str):
            return {
                "valid": False,
                "reason": "Generated output must be text."
            }

        question = question.strip()

        if not question:
            return {
                "valid": False,
                "reason": "Question cannot be empty."
            }

        if len(question) < self.min_length:
            return {
                "valid": False,
                "reason": "Question is too short."
            }

        if len(question) > self.max_length:
            return {
                "valid": False,
                "reason": "Question exceeds maximum length."
            }

        if not question.endswith("?"):
            return {
                "valid": False,
                "reason": "Question must end with a question mark."
            }

        if len(re.findall(r"[a-zA-Z0-9]+", question)) < 3:
            return {
                "valid": False,
                "reason": "Question must contain at least three words."
            }

        normalized_question = re.sub(
            r"\s+",
            " ",
            question.lower()
        ).strip()

        if previous_questions is not None:

            for previous in previous_questions:

                if not isinstance(previous, str):
                    continue

                normalized_previous = re.sub(
                    r"\s+",
                    " ",
                    previous.lower()
                ).strip()

                if normalized_question == normalized_previous:
                    return {
                        "valid": False,
                        "reason": "Duplicate question detected."
                    }

        return {
            "valid": True,
            "reason": "Question is valid.",
            "question": question,
            "interview_category": interview_category,
            "job_role": job_role
        }
