
from Question_Generation.question_engine.interview_context import InterviewContext
from Question_Generation.follow_up_engine.follow_up_generator import FollowUpGenerator
from Question_Generation.validation.question_validator import QuestionValidator


class InterviewQuestionController:

    def __init__(self, question_generator):

        self.question_generator = question_generator

        self.follow_up_generator = FollowUpGenerator(
            question_generator
        )

        self.validator = QuestionValidator()

        self.context = None

    def _validate_and_add_question(
        self,
        question,
        generation_function=None,
        max_attempts=3
    ):

        previous_questions = (
            self.context.get_context()["previous_questions"]
        )

        current_question = question

        for attempt in range(max_attempts):

            result = self.validator.validate(
                current_question,
                previous_questions
            )

            if result["valid"]:

                self.context.add_question(current_question)

                return current_question

            if generation_function is None:
                raise ValueError(
                    f"Question validation failed: {result['reason']}"
                )

            print(
                f"Invalid question. Regeneration attempt {attempt + 1}"
            )

            current_question = generation_function()

        raise RuntimeError(
            "Unable to generate a valid question after maximum attempts."
        )

    def start_interview(
        self,
        interview_category,
        job_role,
        difficulty_level
    ):

        self.context = InterviewContext(
            interview_category=interview_category,
            job_role=job_role,
            difficulty_level=difficulty_level
        )

        question = self.question_generator.generate_question(
            interview_category=interview_category,
            job_role=job_role,
            difficulty_level=difficulty_level,
            previous_questions=[],
            previous_answer=None,
            question_type="initial"
        )

        return self._validate_and_add_question(
            question,
            generation_function=lambda: self.question_generator.generate_question(
                interview_category=interview_category,
                job_role=job_role,
                difficulty_level=difficulty_level,
                previous_questions=[],
                previous_answer=None,
                question_type="initial"
            )
        )

    def submit_answer(self, answer):

        if self.context is None:
            raise RuntimeError("Start the interview first.")

        self.context.add_answer(answer)

        return self.context.get_context()

    def generate_follow_up(self, evaluation_result):

        if self.context is None:
            raise RuntimeError("Start the interview first.")

        if not self.context.question_history:
            raise RuntimeError("No previous answered question found.")

        last_entry = self.context.question_history[-1]

        previous_question = last_entry["question"]
        previous_answer = last_entry["answer"]

        previous_questions = [
            entry["question"]
            for entry in self.context.question_history
        ]

        question = self.follow_up_generator.generate_follow_up(
            interview_category=self.context.interview_category,
            job_role=self.context.job_role,
            difficulty_level=self.context.difficulty_level,
            previous_questions=previous_questions,
            previous_question=previous_question,
            previous_answer=previous_answer,
            evaluation_result=evaluation_result
        )

        return self._validate_and_add_question(
            question,
            generation_function=lambda: self.follow_up_generator.generate_follow_up(
                interview_category=self.context.interview_category,
                job_role=self.context.job_role,
                difficulty_level=self.context.difficulty_level,
                previous_questions=previous_questions,
                previous_question=previous_question,
                previous_answer=previous_answer,
                evaluation_result=evaluation_result
            )
        )

    def generate_next_question(self):

        if self.context is None:
            raise RuntimeError("Start the interview first.")

        context_data = self.context.get_context()

        previous_answer = (
            context_data["previous_answers"][-1]
            if context_data["previous_answers"]
            else None
        )

        question = self.question_generator.generate_question(
            interview_category=self.context.interview_category,
            job_role=self.context.job_role,
            difficulty_level=self.context.difficulty_level,
            previous_questions=context_data["previous_questions"],
            previous_answer=previous_answer,
            question_type="next"
        )

        return self._validate_and_add_question(
            question,
            generation_function=lambda: self.question_generator.generate_question(
                interview_category=self.context.interview_category,
                job_role=self.context.job_role,
                difficulty_level=self.context.difficulty_level,
                previous_questions=context_data["previous_questions"],
                previous_answer=previous_answer,
                question_type="next"
            )
        )

    def get_interview_context(self):

        if self.context is None:
            raise RuntimeError("No active interview.")

        return self.context.get_context()

    def reset_interview(self):

        self.context = None
