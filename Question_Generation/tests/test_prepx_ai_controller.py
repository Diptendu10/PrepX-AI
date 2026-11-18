
from src.prepx_ai.prepx_ai_controller import PrepXAIController


class MockQuestionService:

    def __init__(self):

        self.sessions = {}

    def start_interview(
        self,
        interview_category,
        job_role,
        difficulty_level="medium"
    ):

        session_id = "test-session-001"

        self.sessions[session_id] = {
            "interview_category": interview_category,
            "job_role": job_role,
            "difficulty_level": difficulty_level,
            "question_history": [],
            "current_question": "What is Python?",
            "question_count": 1
        }

        return {
            "session_id": session_id,
            "question": "What is Python?"
        }

    def submit_answer(self, session_id, answer):

        session = self.sessions[session_id]

        session["question_history"].append({
            "question": session["current_question"],
            "answer": answer
        })

        session["current_question"] = None

        return {"message": "Answer submitted successfully."}

    def generate_follow_up(self, session_id):

        session = self.sessions[session_id]

        question = "What are the main features of Python?"

        session["current_question"] = question
        session["question_count"] += 1

        return {"question": question}

    def generate_next_question(self, session_id):

        session = self.sessions[session_id]

        question = "Explain object oriented programming?"

        session["current_question"] = question
        session["question_count"] += 1

        return {"question": question}

    def get_context(self, session_id):

        return self.sessions[session_id].copy()

    def reset_interview(self, session_id):

        del self.sessions[session_id]

        return {"message": "Interview ended successfully."}


def main():

    print("\n========== PREPX-AI CONTROLLER TEST ==========\n")

    mock_service = MockQuestionService()

    controller = PrepXAIController(
        question_service=mock_service
    )

    # Step 1: Start interview
    result = controller.start_interview(
        interview_category="Software Development",
        job_role="Python Developer",
        difficulty_level="medium"
    )

    session_id = result["session_id"]

    print("Interview Started")
    print("Session ID:", session_id)
    print("AI:", result["question"])

    # Step 2: Submit candidate answer
    answer = (
        "Python is a high-level programming language "
        "known for readability and simplicity."
    )

    result = controller.submit_candidate_answer(
        session_id,
        answer
    )

    print("\nCandidate:", answer)
    print("System:", result["message"])

    # Step 3: Generate follow-up
    result = controller.generate_follow_up(session_id)

    print("\nAI Follow-up:", result["question"])

    # Step 4: Submit another answer
    second_answer = (
        "Python supports object-oriented, procedural "
        "and functional programming."
    )

    controller.submit_candidate_answer(
        session_id,
        second_answer
    )

    print("\nCandidate:", second_answer)

    # Step 5: Generate next question
    result = controller.generate_next_question(session_id)

    print("\nAI Next Question:", result["question"])

    # Step 6: Get interview context
    context = controller.get_interview_context(session_id)

    print("\nInterview Context:")
    print("Category:", context["interview_category"])
    print("Role:", context["job_role"])
    print("Difficulty:", context["difficulty_level"])
    print("Questions Asked:", context["question_count"])

    # Step 7: End interview
    result = controller.end_interview(session_id)

    print("\n", result["message"])

    print("\nAll Controller Tests Passed!")


if __name__ == "__main__":
    main()
