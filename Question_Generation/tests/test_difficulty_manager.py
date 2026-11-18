
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from Question_Generation.difficulty_control.difficulty_manager import DifficultyManager


# Step 1: Create Difficulty Manager
manager = DifficultyManager()

# Step 2: Get available levels
levels = manager.get_available_levels()

print("\n========== PREPX AI ==========")

print("\nAvailable Difficulty Levels:")

for level in levels:
    print(level.capitalize())

# Step 3: Test each difficulty
for level in ["easy", "medium", "hard"]:

    instruction = manager.get_difficulty_instruction(level)

    print(f"\n{level.upper()} DIFFICULTY:")
    print(instruction)

# Step 4: Validate difficulty
assert manager.validate_difficulty("Easy") == "easy"
assert manager.validate_difficulty("MEDIUM") == "medium"
assert manager.validate_difficulty("Hard") == "hard"

# Step 5: Test invalid difficulty
try:

    manager.validate_difficulty("Very Hard")

    raise AssertionError("Invalid difficulty was accepted")

except ValueError:
    print("\nInvalid Difficulty Rejected Successfully!")

print("\nDifficulty Manager Working Successfully!")
