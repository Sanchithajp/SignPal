# main.py

from fastapi import FastAPI, UploadFile, File, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import requests
import re
import os
from dotenv import load_dotenv
import os



import models
from database import SessionLocal, engine
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY not set in .env")
# ----- Load Keras model -----
MODEL_PATH = "asl_model.h5"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)

# Labels for A-Z
label_map = [chr(i) for i in range(65, 91)]

# Create DB tables
models.Base.metadata.create_all(bind=engine)

# ----- FastAPI setup -----
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----- Pydantic Schemas -----
class UserRequest(BaseModel):
    username: str
    password: str

# ----- Auth endpoints -----

@app.post("/signup")
def signup(user: UserRequest, db: Session = Depends(get_db)):
    existing_user = db.query(models.User).filter(models.User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists.")
    new_user = models.User(username=user.username, password=user.password)
    db.add(new_user)
    db.commit()
    return {"success": True, "message": "User created."}


@app.post("/login")
def login(user: UserRequest, db: Session = Depends(get_db)):
    user_obj = db.query(models.User).filter(
        models.User.username == user.username,
        models.User.password == user.password
    ).first()
    if not user_obj:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    return {"success": True, "message": "Logged in."}

# ----- Predict endpoint -----

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

        # Convert to grayscale and resize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        normalized = resized / 255.0
        input_tensor = normalized.reshape(1, 28, 28, 1)

        preds = model.predict(input_tensor, verbose=0)
        pred_index = int(np.argmax(preds))
        letter = label_map[pred_index] if pred_index < len(label_map) else "?"

        return {
            "letter": letter,
            "probs": preds.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----- Explain endpoint -----

@app.post("/explain")
async def explain(request: Request):
    data = await request.json()
    word = data.get("word", "")

    prompt = f"""
    Explain the term '{word}' in simple language suitable for kids.
    Provide a YouTube link showing how to sign the word '{word}' in Indian Sign Language (ISL).
    If there is no ISL video, suggest any sign language video (e.g. ASL, BSL) demonstrating this word.
    """

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable not set.")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {groq_api_key}"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful educational assistant who explains things simply for children."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.4
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"]

        # Find YouTube link in answer
        urls = re.findall(r"(https?://[^\s]+)", answer)
        yt_link = urls[0] if urls else None

        return {
            "answer": answer,
            "youtube_link": yt_link
        }

    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error contacting Groq API: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
