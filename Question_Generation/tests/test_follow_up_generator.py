
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Question_Generation.follow_up_engine.follow_up_generator import FollowUpGenerator


# Mock Question Generator
class MockQuestionGenerator:

    def generate_question(self, **kwargs):

        assert kwargs["question_type"] == "follow_up"

        assert kwargs["previous_answer"] is not None

        assert kwargs["evaluation_result"] is not None

        return "Can you explain why you chose that approach?"


# Step 1: Create Follow-up Generator
question_generator = MockQuestionGenerator()

follow_up_generator = FollowUpGenerator(
    question_generator
)

# Step 2: Prepare interview data
evaluation_result = {
    "relevance_score": 8,
    "correctness_score": 6,
    "completeness_score": 5,
    "feedback": "The answer needs more explanation."
}

# Step 3: Generate follow-up
follow_up_question = follow_up_generator.generate_follow_up(

    interview_category="Technical",

    job_role="Python Developer",

    difficulty_level="Medium",

    previous_questions=[
        "What is Python?"
    ],

    previous_question="What is Python?",

    previous_answer="Python is a programming language.",

    evaluation_result=evaluation_result
)

# Step 4: Display result
print("\n========== PREPX AI ==========")

print("\nPrevious Question:")
print("What is Python?")

print("\nCandidate Answer:")
print("Python is a programming language.")

print("\nEvaluation Result:")
print(evaluation_result)

print("\nGenerated Follow-up:")
print(follow_up_question)

# Step 5: Validate
assert isinstance(follow_up_question, str)
assert len(follow_up_question) > 0

print("\nFollow-up Question Engine Working Successfully!")

