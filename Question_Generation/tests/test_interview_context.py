
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Question_Generation.question_engine.interview_context import InterviewContext


# Step 1: Create context manager
context = InterviewContext()

# Step 2: Start interview
context.start_interview(
    interview_category="Technical",
    job_role="Python Developer",
    difficulty_level="Medium"
)

# Step 3: Add first question
context.add_question(
    "What is Python?"
)

# Step 4: Add candidate answer
context.add_answer(
    "Python is a high-level programming language."
)

# Step 5: Add second question
context.add_question(
    "What are Python's main features?"
)

# Step 6: Add second answer
context.add_answer(
    "Python supports object-oriented programming."
)

# Step 7: Get interview context
interview_data = context.get_context()

print("\n========== PREPX AI ==========")

print("\nInterview Configuration:")
print("Category:", interview_data["interview_category"])
print("Job Role:", interview_data["job_role"])
print("Difficulty:", interview_data["difficulty_level"])

print("\nQuestion Count:")
print(interview_data["question_count"])

print("\nInterview History:")

for item in interview_data["question_history"]:
    print("\nQuestion:", item["question"])
    print("Answer:", item["answer"])

# Step 8: Validate
assert interview_data["question_count"] == 2
assert len(interview_data["previous_questions"]) == 2
assert len(interview_data["previous_answers"]) == 2

print("\nInterview Context Working Successfully!")
