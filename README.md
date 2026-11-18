# PrepX-AI

PrepX-AI is a Python-based interview question generation system.

## Run the API

```powershell
pip install -r requirements.txt
python -m uvicorn Integration_API.question_generation_api:app --reload
```

## Run tests

```powershell
pytest Question_Generation/tests
```