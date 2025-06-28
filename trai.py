from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import cv2
import os

# Create FastAPI app
app = FastAPI()

# Allow requests from frontend running locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # you can restrict to ["http://localhost:5500"] etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = tf.keras.models.load_model("asl_model.h5")

# Mapping label index → alphabet letter
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded image temporarily
        contents = await file.read()
        temp_path = "temp.jpg"
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Read as grayscale
        img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"error": "Image could not be read."}

        # Resize to 28x28 as per training data
        img = cv2.resize(img, (28, 28))

        # Normalize
        img = img / 255.0

        # Add channels and batch dimensions
        img = np.expand_dims(img, axis=-1)   # shape → (28, 28, 1)
        img = np.expand_dims(img, axis=0)    # shape → (1, 28, 28, 1)

        # Predict
        pred_probs = model.predict(img, verbose=0)
        pred_index = np.argmax(pred_probs)
        predicted_letter = alphabet[pred_index]

        print("Probabilities:", pred_probs)
        print("Predicted:", predicted_letter)

        return {
            "letter": predicted_letter,
            "probs": pred_probs.tolist()
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/explain")
async def explain(data: dict):
    """
    Dummy endpoint for your AI explanation
    """
    word = data.get("word", "")
    return {
        "answer": f"You typed: {word}. (Replace this with Groq API call if you want.)",
        "youtube_link": None
    }
