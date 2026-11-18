
from Question_Generation.difficulty_control.difficulty_manager import (
    DifficultyManager
)

from Question_Generation.config.model_config import ModelConfig


class QuestionGenerator:

    def __init__(self, text_generator):

        self.text_generator = text_generator
        self.difficulty_manager = DifficultyManager()

    def generate_question(
        self,
        interview_category,
        job_role,
        difficulty_level,
        previous_questions=None,
        previous_answer=None,
        question_type="initial",
        evaluation_result=None,
        previous_answers=None,
        question_history=None
    ):

        # Validate interview category.
        if not isinstance(interview_category, str):
            raise ValueError("Interview category must be text.")

        if not interview_category.strip():
            raise ValueError("Interview category cannot be empty.")

        # Validate job role.
        if not isinstance(job_role, str):
            raise ValueError("Job role must be text.")

        if not job_role.strip():
            raise ValueError("Job role cannot be empty.")

        # Validate question type.
        supported_types = [
            "initial",
            "follow_up",
            "next"
        ]

        if not isinstance(question_type, str):
            raise ValueError("Question type must be text.")

        question_type = question_type.strip().lower()

        if question_type not in supported_types:
            raise ValueError(
                f"Unsupported question type: {question_type}"
            )

        # Normalize difficulty.
        difficulty_level = (
            self.difficulty_manager.normalize_difficulty(
                difficulty_level
            )
        )

        difficulty_instruction = (
            self.difficulty_manager.get_difficulty_instruction(
                difficulty_level
            )
        )

        # Validate previous questions.
        if previous_questions is None:
            previous_questions = []

        if not isinstance(previous_questions, list):
            raise ValueError("Previous questions must be a list.")

        for question in previous_questions:
            if not isinstance(question, str):
                raise ValueError(
                    "Every previous question must be text."
                )

        previous_questions_text = (
            "\n".join(
                f"- {question}"
                for question in previous_questions
            )
            if previous_questions
            else "No previous questions."
        )

        # Validate previous answers.
        if previous_answers is None:
            previous_answers = []

        if not isinstance(previous_answers, list):
            raise ValueError("Previous answers must be a list.")

        for answer in previous_answers:
            if not isinstance(answer, str):
                raise ValueError(
                    "Every previous answer must be text."
                )

        previous_answers_text = (
            "\n".join(
                f"- {answer}"
                for answer in previous_answers
            )
            if previous_answers
            else "No previous answers."
        )

        # Prepare latest answer.
        if previous_answer is None:
            previous_answer_text = "No previous answer."

        elif isinstance(previous_answer, str):
            previous_answer_text = previous_answer.strip()

            if not previous_answer_text:
                previous_answer_text = "No previous answer."

        else:
            raise ValueError("Previous answer must be text.")

        # Prepare complete interview history.
        if question_history is None:
            question_history = []

        if not isinstance(question_history, list):
            raise ValueError("Question history must be a list.")

        history_lines = []

        for index, entry in enumerate(question_history, start=1):

            if not isinstance(entry, dict):
                raise ValueError(
                    "Each history entry must be a dictionary."
                )

            history_question = entry.get("question")
            history_answer = entry.get("answer")

            if not isinstance(history_question, str):
                raise ValueError(
                    "History question must be text."
                )

            if not isinstance(history_answer, str):
                raise ValueError(
                    "History answer must be text."
                )

            history_lines.append(
                f"Question {index}: {history_question}\n"
                f"Candidate Answer {index}: {history_answer}"
            )

        history_text = (
            "\n\n".join(history_lines)
            if history_lines
            else "No completed interview conversation."
        )

        # Prepare evaluation information.
        if evaluation_result is None:
            evaluation_text = "No evaluation available."

        elif isinstance(evaluation_result, str):
            evaluation_text = evaluation_result

        else:
            evaluation_text = str(evaluation_result)

        # Build context-aware prompt.
        prompt = f"""
You are PrepX-AI, an interview question generation system.

Your task is to generate exactly ONE interview question.

INTERVIEW INFORMATION:

Interview Category: {interview_category}
Job Role: {job_role}
Difficulty: {difficulty_level}

DIFFICULTY INSTRUCTIONS:

{difficulty_instruction}

QUESTION TYPE:

{question_type}

PREVIOUS QUESTIONS:

{previous_questions_text}

PREVIOUS ANSWERS:

{previous_answers_text}

COMPLETE INTERVIEW HISTORY:

{history_text}

LATEST CANDIDATE ANSWER:

{previous_answer_text}

LATEST ANSWER EVALUATION:

{evaluation_text}

GENERATION RULES:

1. Generate exactly one interview question.
2. The question must be relevant to the selected job role.
3. Follow the selected interview category.
4. Follow the selected difficulty level.
5. Consider the candidate's previous answers.
6. Do not repeat previous questions.
7. Use interview history to maintain conversational continuity.
8. Do not generate an answer or explanation.
9. Do not include question numbers or headings.
10. End the generated question with a question mark.

QUESTION TYPE RULES:

- initial:
  Generate the first interview question.
  Do not assume the candidate has answered anything.

- follow_up:
  Generate a relevant follow-up based on the latest candidate answer.
  Explore the candidate's understanding further.

- next:
  Generate a new question about a different relevant topic.
  Avoid repeating previously covered questions.

Return only the interview question.

QUESTION:
"""

        # Generate question.
        question = self.text_generator.generate(
            prompt=prompt,
            max_new_tokens=ModelConfig.MAX_NEW_TOKENS,
            temperature=ModelConfig.TEMPERATURE
        )

        if not isinstance(question, str):
            raise ValueError(
                "Question generator must return text."
            )

        question = question.strip()

        if not question:
            raise ValueError(
                "Generated question cannot be empty."
            )

        return question
