
from Question_Generation.config.model_config import ModelConfig


class DifficultyManager:

    DIFFICULTY_INSTRUCTIONS = {

        "easy": (
            "Generate a beginner-friendly interview question. "
            "Focus on basic definitions, fundamental concepts "
            "and simple explanations. Avoid advanced terminology."
        ),

        "medium": (
            "Generate an intermediate interview question. "
            "Focus on practical applications, comparisons, "
            "technical explanations and problem-solving."
        ),

        "hard": (
            "Generate an advanced interview question. "
            "Focus on complex concepts, analytical reasoning, "
            "system design and challenging real-world scenarios."
        )
    }

    def __init__(self):
        self.supported_difficulties = (
            ModelConfig.SUPPORTED_DIFFICULTIES
        )

        self.default_difficulty = (
            ModelConfig.DEFAULT_DIFFICULTY
        )

    def normalize_difficulty(self, difficulty_level):

        if not isinstance(difficulty_level, str):
            raise ValueError(
                "Difficulty level must be a string."
            )

        difficulty_level = difficulty_level.strip().lower()

        if difficulty_level not in self.supported_difficulties:
            raise ValueError(
                f"Unsupported difficulty: {difficulty_level}. "
                f"Choose from {self.supported_difficulties}."
            )

        return difficulty_level

    def get_difficulty_instruction(self, difficulty_level):

        difficulty_level = self.normalize_difficulty(
            difficulty_level
        )

        return self.DIFFICULTY_INSTRUCTIONS[difficulty_level]

    def get_default_difficulty(self):
        return self.default_difficulty

    def get_supported_difficulties(self):
        return self.supported_difficulties.copy()
