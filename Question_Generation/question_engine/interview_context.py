
class InterviewContext:

    def __init__(
        self,
        interview_category="General",
        job_role="General",
        difficulty_level="medium"
    ):

        self.interview_category = interview_category
        self.job_role = job_role
        self.difficulty_level = difficulty_level

        self.question_history = []
        self.current_question = None
        self.question_count = 0

    def start_interview(
        self,
        interview_category="General",
        job_role="General",
        difficulty_level="medium"
    ):

        self.interview_category = interview_category
        self.job_role = job_role
        self.difficulty_level = difficulty_level

        self.reset()

    def add_question(self, question):

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        if self.current_question is not None:
            raise ValueError(
                "The previous question must be answered before adding another."
            )

        self.current_question = question.strip()
        self.question_count += 1

    def add_answer(self, answer):

        if self.current_question is None:
            raise ValueError(
                "There is no pending question to answer."
            )

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Answer must be a non-empty string.")

        self.question_history.append({
            "question": self.current_question,
            "answer": answer.strip()
        })

        self.current_question = None

    def get_context(self):

        history = [
            {
                "question": entry["question"],
                "answer": entry["answer"]
            }
            for entry in self.question_history
        ]

        previous_questions = [
            entry["question"]
            for entry in history
        ]

        previous_answers = [
            entry["answer"]
            for entry in history
        ]

        return {
            "interview_category": self.interview_category,
            "job_role": self.job_role,
            "difficulty_level": self.difficulty_level,
            "question_history": history,
            "previous_questions": previous_questions,
            "previous_answers": previous_answers,
            "current_question": self.current_question,
            "question_count": self.question_count
        }

    def get_latest_answer(self):

        if not self.question_history:
            return None

        return self.question_history[-1]["answer"]

    def get_latest_question(self):

        if not self.question_history:
            return None

        return self.question_history[-1]["question"]

    def reset(self):

        self.question_history = []
        self.current_question = None
        self.question_count = 0
