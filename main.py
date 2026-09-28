import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.5-flash-lite"


class TextRequest(BaseModel):
    text: str


class QARequest(BaseModel):
    question: str


@app.get("/")
def home():
    return FileResponse("frontend/index.html")


# 1. Q&A
@app.post("/qa")
def qa(request: QARequest):
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Answer this educational question clearly and concisely:\n{request.question}"
    )

    return {
        "question": request.question,
        "answer": response.text
    }


# 2. Explanation
@app.post("/explain")
def explain(request: TextRequest):
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Explain the following topic in simple, beginner-friendly language:\n{request.text}"
    )

    return {
        "topic": request.text,
        "explanation": response.text
    }


# 3. Quiz
@app.post("/quiz")
def quiz(request: TextRequest):
    response = client.models.generate_content(
        model=MODEL,
        contents=f"""
Create a quiz about this topic: {request.text}

Generate exactly 3 multiple-choice questions.
Each question must have exactly 4 options.
Clearly mention the correct answer.
Keep it suitable for students.
"""
    )

    return {
        "topic": request.text,
        "quiz": response.text
    }


# 4. Summarize
@app.post("/summarize")
def summarize(request: TextRequest):
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Summarize the following educational content clearly and concisely:\n{request.text}"
    )

    return {
        "summary": response.text
    }


# 5. Learning Recommendations
@app.post("/learn/recommendations")
def learning_recommendations(request: TextRequest):
    response = client.models.generate_content(
        model=MODEL,
        contents=f"""
Create a structured learning path for: {request.text}

Organize it from beginner to advanced level.
Include useful learning resources such as videos, articles, or books.
Give step-by-step guidance.
"""
    )

    return {
        "topic": request.text,
        "recommendations": response.text
    }
