
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.prepx_ai.question_generation_service import (
    QuestionGenerationService
)


app = FastAPI(
    title="PrepX-AI Question Generation API",
    description="API for AI-powered interview question generation.",
    version="1.0.0"
)

service = QuestionGenerationService()


class StartInterviewRequest(BaseModel):
    interview_category: str = Field(min_length=1)
    job_role: str = Field(min_length=1)
    difficulty_level: str = "medium"


class AnswerRequest(BaseModel):
    session_id: str
    answer: str = Field(min_length=1)


class SessionRequest(BaseModel):
    session_id: str


@app.get("/")
def home():

    return {
        "message": "Welcome to PrepX-AI Question Generation API"
    }


@app.post("/question/start")
def start_interview(request: StartInterviewRequest):

    try:

        return service.start_interview(
            interview_category=request.interview_category,
            job_role=request.job_role,
            difficulty_level=request.difficulty_level
        )

    except (ValueError, RuntimeError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


@app.post("/question/answer")
def submit_answer(request: AnswerRequest):

    try:

        return service.submit_answer(
            request.session_id,
            request.answer
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


@app.post("/question/follow-up")
def generate_follow_up(request: SessionRequest):

    try:

        return service.generate_follow_up(
            request.session_id
        )

    except (ValueError, RuntimeError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


@app.post("/question/next")
def generate_next_question(request: SessionRequest):

    try:

        return service.generate_next_question(
            request.session_id
        )

    except (ValueError, RuntimeError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


@app.get("/question/context/{session_id}")
def get_context(session_id: str):

    try:

        return service.get_context(session_id)

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )


@app.post("/question/reset")
def reset_interview(request: SessionRequest):

    try:

        return service.reset_interview(
            request.session_id
        )

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )
