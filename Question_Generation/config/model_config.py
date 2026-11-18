
class ModelConfig:

    # Model architecture
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 64
    NUM_HEADS = 4
    NUM_LAYERS = 2
    HIDDEN_DIM = 128
    MAX_SEQUENCE_LENGTH = 128

    # Text generation
    MAX_NEW_TOKENS = 30
    TEMPERATURE = 0.8

    # Question validation
    MIN_QUESTION_LENGTH = 10
    MAX_QUESTION_LENGTH = 300

    # Question regeneration
    MAX_REGENERATION_ATTEMPTS = 3

    # Interview configuration
    DEFAULT_DIFFICULTY = "medium"

    SUPPORTED_DIFFICULTIES = [
        "easy",
        "medium",
        "hard"
    ]
