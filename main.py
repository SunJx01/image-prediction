# import os
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse, StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# from io import BytesIO
# from PIL import Image
# from utils import (
#     get_result_image,
#     get_result_image_tl,
#     load_preprocess_image_cnn,
#     load_preprocess_image_tl,
#     predict_tl,
#     get_model
# )

# MODEL_PATH = "trained_model.h5"

# app = FastAPI()

# # Allow requests from frontend (React app)
# origins = [
#     "http://localhost:3000",  # Frontend URL
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,  # List of allowed origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all methods
#     allow_headers=["*"],  # Allow all headers
# )

# @app.get("/train-model")
# async def train_model():
#     if not os.path.exists(MODEL_PATH):
#         # Train model logic (you can call your existing `get_model` function here)
#         get_model()
#         return JSONResponse(content={"message": "Model trained and saved."}, status_code=200)
#     return JSONResponse(content={"message": "Model already exists, no need to train."}, status_code=200)


# @app.post("/upload-image")
# async def upload_image(File: UploadFile = File(...)):
#     # Read the file bytes
#     image_bytes = await File.read()

#     img_pil = Image.open(BytesIO(image_bytes))

#     # Ensure the file is a valid image
#     if File.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
#         raise HTTPException(
#             status_code=400,
#             detail="Invalid image format. Only PNG, JPG, and JPEG are allowed.",
#         )

#     # Load and preprocess the image
#     preprocessed_img = load_preprocess_image_tl(img_pil)

#     # Get prediction result
#     prediction_result = predict_tl(preprocessed_img)

#     # Generate result image (this could return a processed image as bytes)
#     result_img_bytes = get_result_image_tl(img_pil, prediction_result)

#     # Return the processed image as a streaming response
#     return StreamingResponse(result_img_bytes, media_type="image/png")


import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from utils import (
    get_result_image_tl,
    load_preprocess_image_tl,
    predict_tl,
    get_model
)

# Path to save/load the trained model
MODEL_PATH = "trained_model.h5"

app = FastAPI()

# Define allowed origins for CORS (e.g., frontend URL)
origins = [
    "http://localhost:3000",  # Replace with your frontend URL
]

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
    if not os.path.exists(MODEL_PATH):
        get_model()  # Train and save the model
        return JSONResponse(content={"message": "Model trained and saved."}, status_code=200)
    return JSONResponse(content={"message": "Model already exists, no need to train."}, status_code=200)

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint to upload an image, preprocess it, predict the class,
    and return the result image with the prediction.
    """
    # Check if the uploaded file is a valid image format
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Only PNG, JPG, and JPEG are allowed."
        )

    # Read image bytes and open as PIL image
    image_bytes = await file.read()
    img_pil = Image.open(BytesIO(image_bytes))

    # Preprocess the image for prediction
    preprocessed_img = load_preprocess_image_tl(img_pil)

    # Predict the class of the image
    prediction_result = predict_tl(preprocessed_img)

    # Generate result image with the prediction label
    result_img_bytes = get_result_image_tl(img_pil, prediction_result)

    # Return the result image as a StreamingResponse
    return StreamingResponse(result_img_bytes, media_type="image/png")
