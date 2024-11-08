import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from utils import (
    display_prediction,
    load_preprocess_image,
    predict_image,
    build_and_save_model,
)

# Path to save/load the trained model
MODEL_PATH = "trained_model.h5"

app = FastAPI()

# Define allowed origins for CORS (e.g., frontend URL)
origins = ["*"]

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/train-model")
async def train_model():
    """
    Train and save the model if it does not already exist.
    """
    print("heheh")
    if not os.path.exists(MODEL_PATH):
        build_and_save_model()  # Train and save the model
        return JSONResponse(
            content={"message": "Model trained and saved."}, status_code=200
        )
    return JSONResponse(
        content={"message": "Model already exists, no need to train."}, status_code=200
    )


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint to upload an image, preprocess it, predict the class,
    and return the result image with the prediction.
    """
    print("bhitra ta aayo hola")
    # Check if the uploaded file is a valid image format
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Only PNG, JPG, and JPEG are allowed.",
        )
    print("eta chai?")

    # Read image bytes and open as PIL image
    image_bytes = await file.read()
    img_pil = Image.open(BytesIO(image_bytes))

    # Check the type of img_pil before passing it
    if not isinstance(img_pil, Image.Image):
        print("Expected a PIL Image object")
        raise TypeError("Expected a PIL Image object")
    print(type(img_pil), "typeee1")
    # Preprocess the image for prediction
    # preprocessed_img = load_preprocess_image(img_pil)

    # Predict the class of the image
    prediction_result = predict_image(img_pil)

    # Generate result image with the prediction label
    result_img_bytes = display_prediction(img_pil, prediction_result)

    # Return the result image as a StreamingResponse
    return StreamingResponse(result_img_bytes, media_type="image/png")
