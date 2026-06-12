"""
Improved Room Classification API
Features:
- Support for both original and improved models
- Confidence scores in predictions
- Batch prediction support
- Better error handling
- Health check endpoint
- Model info endpoint
- CORS support
- Request validation
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import uvicorn
from typing import List, Optional
import os
from datetime import datetime

# Configuration
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Load model - try improved model first, fallback to original
def load_model():
    """Load the best available model"""
    model_paths = [
        'trained_model_improved',
        'saved_model_improved',
        'trained_model'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"Loading model from: {path}")
            return tf.keras.models.load_model(path), path
    
    raise FileNotFoundError("No trained model found. Please train a model first.")

try:
    MODEL, MODEL_PATH = load_model()
    IMG_SIZE = 224 if 'improved' in MODEL_PATH else 150
    print(f"Model loaded successfully! Image size: {IMG_SIZE}x{IMG_SIZE}")
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL = None
    MODEL_PATH = None
    IMG_SIZE = 224

app = FastAPI(
    title="Room Classification API",
    description="API untuk mengklasifikasikan apakah kamar rapi atau berantakan",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_file(file: UploadFile) -> bool:
    """Validate uploaded file"""
    if not file.filename:
        return False
    
    # Check file extension
    ext = file.filename.split('.')[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False
    
    return True

def preprocess_image(content: io.BytesIO, img_size: int = IMG_SIZE):
    """Preprocess image for prediction"""
    img = image.load_img(content, target_size=(img_size, img_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    # Normalize if using improved model
    if 'improved' in str(MODEL_PATH):
        x = x / 255.0
    
    return x

def predict_image(img_array) -> dict:
    """Make prediction and return detailed results"""
    prediction = MODEL.predict(img_array, verbose=0)[0][0]
    
    # Determine class and confidence
    if prediction > 0.5:
        predicted_class = "messy"
        label_indonesian = "Kamar berantakan"
        confidence = float(prediction)
    else:
        predicted_class = "clean"
        label_indonesian = "Kamar rapi"
        confidence = float(1 - prediction)
    
    return {
        "predicted_class": predicted_class,
        "label_indonesian": label_indonesian,
        "confidence": round(confidence * 100, 2),
        "probability": {
            "messy": round(float(prediction) * 100, 2),
            "clean": round(float(1 - prediction) * 100, 2)
        }
    }

@app.get('/')
async def index():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Room Classification API",
        "version": "2.0.0",
        "endpoints": {
            "/": "This information page",
            "/health": "Health check",
            "/model/info": "Model information",
            "/predict": "Single image prediction (POST)",
            "/predict/batch": "Batch image prediction (POST)"
        },
        "model_path": MODEL_PATH,
        "image_size": f"{IMG_SIZE}x{IMG_SIZE}"
    }

@app.get('/health')
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if MODEL is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "timestamp": datetime.now().isoformat()
    }

@app.get('/model/info')
async def model_info():
    """Get model information"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": MODEL_PATH,
        "image_size": f"{IMG_SIZE}x{IMG_SIZE}",
        "input_shape": MODEL.input_shape,
        "output_shape": MODEL.output_shape,
        "total_params": MODEL.count_params(),
        "classes": ["clean", "messy"]
    }

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """
    Predict whether a room is clean or messy from an uploaded image
    
    Returns:
        JSON with prediction results including confidence scores
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file
    if not validate_file(file):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    try:
        # Read and process image
        upload = await file.read()
        
        # Check file size
        if len(upload) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        content = io.BytesIO(upload)
        img_array = preprocess_image(content)
        
        # Make prediction
        result = predict_image(img_array)
        
        # Add metadata
        result["filename"] = file.filename
        result["timestamp"] = datetime.now().isoformat()
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post('/predict/batch')
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict multiple images at once
    
    Returns:
        JSON with prediction results for all images
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    errors = []
    
    for file in files:
        try:
            if not validate_file(file):
                errors.append({
                    "filename": file.filename,
                    "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
                })
                continue
            
            upload = await file.read()
            
            if len(upload) > MAX_FILE_SIZE:
                errors.append({
                    "filename": file.filename,
                    "error": f"File too large. Maximum: {MAX_FILE_SIZE // (1024*1024)}MB"
                })
                continue
            
            content = io.BytesIO(upload)
            img_array = preprocess_image(content)
            result = predict_image(img_array)
            result["filename"] = file.filename
            results.append(result)
            
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(
        content={
            "successful_predictions": len(results),
            "errors": len(errors),
            "results": results,
            "errors_detail": errors if errors else None,
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == '__main__':
    print("="*60)
    print("ROOM CLASSIFICATION API SERVER")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print("\nStarting server...")
    print("Access the API at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("="*60)
    
    uvicorn.run(app, host='0.0.0.0', port=8000)