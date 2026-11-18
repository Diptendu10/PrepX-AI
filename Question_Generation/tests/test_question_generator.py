
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Question_Generation.core.tokenizer import Tokenizer
from Question_Generation.core.language_model import LanguageModel
from Question_Generation.core.generation import TextGenerator

from Question_Generation.question_engine.question_generator import (
    QuestionGenerator
)

from Question_Generation.validation.question_validator import (
    QuestionValidator
)


def main():

    print("\n========== PREPX AI ==========")

    # Step 1: Prepare sample vocabulary.
    texts = [
        "What is Python?",
        "Explain machine learning.",
        "What is artificial intelligence?",
        "Explain neural networks.",
        "What is object oriented programming?",
        "Describe database management.",
        "What are Python decorators?",
        "Explain database indexing.",
        "How does memory management work?",
        "What is the difference between TCP and UDP?"
    ]

    tokenizer = Tokenizer()
    tokenizer.build_vocab(texts)

    print("\n[1] Tokenizer initialized successfully.")

    # Step 2: Initialize Question Generation Model.
    model = LanguageModel(
        vocab_size=len(tokenizer.get_vocab()),
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=64,
        max_length=128
    )

    model.eval()

    print("[2] Language Model initialized successfully.")

    # Step 3: Initialize Text Generator.
    text_generator = TextGenerator(
        model=model,
        tokenizer=tokenizer
    )

    print("[3] Text Generator initialized successfully.")

    # Step 4: Initialize Question Generator.
    question_generator = QuestionGenerator(
        text_generator=text_generator
    )

    print("[4] Question Generator initialized successfully.")

    # Step 5: Generate interview question.
    interview_category = "Technical"
    job_role = "Python Developer"
    difficulty_level = "Medium"

    question = question_generator.generate_question(
        interview_category=interview_category,
        job_role=job_role,
        difficulty_level=difficulty_level,
        previous_questions=[],
        previous_answer=None,
        question_type="initial",
        evaluation_result=None
    )

    print("[5] Question generated successfully.")

    # Step 6: Validate generated question.
    validator = QuestionValidator()

    validation_result = validator.validate(
        question=question,
        previous_questions=[]
    )

    print("\n========== INTERVIEW DETAILS ==========")

    print(f"Interview Category: {interview_category}")
    print(f"Job Role: {job_role}")
    print(f"Difficulty: {difficulty_level}")

    print("\n========== GENERATED QUESTION ==========")

    print(question)

    print("\n========== VALIDATION RESULT ==========")

    print(f"Valid: {validation_result['valid']}")
    print(f"Reason: {validation_result['reason']}")

    print("\n========== MODEL STATUS ==========")

    print(f"Device: {next(model.parameters()).device}")
    print(f"Vocabulary Size: {len(tokenizer.get_vocab())}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")

    print("\nQuestion Generation Engine Integration Test Completed!")


if __name__ == "__main__":
    main()
