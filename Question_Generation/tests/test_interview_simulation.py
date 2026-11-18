
from Question_Generation.question_engine.interview_context import InterviewContext
from Question_Generation.question_engine.question_generator import QuestionGenerator
from Question_Generation.follow_up_engine.follow_up_generator import FollowUpGenerator


class MockTextGenerator:

    def __init__(self):
        self.questions = [
            "What is Python?",
            "What are the main features of Python?",
            "How does object oriented programming work?"
        ]
        self.index = 0

    def generate(self, prompt, max_new_tokens=None, temperature=None):
        if self.index >= len(self.questions):
            return "Can you explain another programming concept?"

        question = self.questions[self.index]
        self.index += 1
        return question


def main():

    print("\nStarting Interview Simulation...\n")

    # Initialize the mock text generator
    text_generator = MockTextGenerator()

    # Initialize question generator
    question_generator = QuestionGenerator(text_generator)

    # Initialize follow-up generator
    follow_up_generator = FollowUpGenerator(question_generator)

    # Create interview context
    context = InterviewContext(
        interview_category="Software Development",
        job_role="Python Developer",
        difficulty_level="medium"
    )

    # Generate initial question
    first_question = question_generator.generate_question(
        interview_category=context.interview_category,
        job_role=context.job_role,
        difficulty_level=context.difficulty_level,
        question_type="initial"
    )

    context.add_question(first_question)

    print("AI:", first_question)

    # Candidate's first answer
    first_answer = (
        "Python is a high-level programming language "
        "known for its readability and simplicity."
    )

    print("\nCandidate:", first_answer)

    context.add_answer(first_answer)

    # Generate follow-up question
    follow_up_question = follow_up_generator.generate_follow_up(
        interview_category=context.interview_category,
        job_role=context.job_role,
        difficulty_level=context.difficulty_level,
        previous_question=first_question,
        previous_answer=first_answer,
        question_history=context.question_history
    )

    context.add_question(follow_up_question)

    print("\nAI Follow-up:", follow_up_question)

    # Candidate's second answer
    second_answer = (
        "Python supports object-oriented, procedural "
        "and functional programming."
    )

    print("\nCandidate:", second_answer)

    context.add_answer(second_answer)

    # Generate next question
    next_question = question_generator.generate_question(
        interview_category=context.interview_category,
        job_role=context.job_role,
        difficulty_level=context.difficulty_level,
        question_history=context.question_history,
        question_type="next"
    )

    context.add_question(next_question)

    print("\nAI Next Question:", next_question)

    # Display interview summary
    print("\n========== INTERVIEW SUMMARY ==========")

    interview_data = context.get_context()

    print("Category:", interview_data["interview_category"])
    print("Job Role:", interview_data["job_role"])
    print("Difficulty:", interview_data["difficulty_level"])
    print("Questions Asked:", interview_data["question_count"])

    print("\nQuestion History:")

    for index, item in enumerate(
        interview_data["question_history"], start=1
    ):
        print(f"\nQuestion {index}: {item['question']}")
        print(f"Answer {index}: {item['answer']}")

    print("\nInterview Simulation Completed!")


if __name__ == "__main__":
    main()

