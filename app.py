from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import os
from typing import List
import asyncio
import uuid
import utils.download_model as da
from utils.image_processor import preprocess_image

app = FastAPI(title="Fish Gender Classification", version="1.0.0")

da.download()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration
IMG_SIZE_X = 384
IMG_SIZE_Y = 250
CONFIDENCE_THRESHOLD = 0.5
DATASET_DIR = "static/dataset"


# Predefined dataset information
PREDEFINED_DATASET = [
    {
        "id": "Фото 1",
        "filename": "f1.jpg",
        "expected_gender": "Female",  # For demo purposes
        "path": "/static/dataset/f1.jpg"
    },
    {
        "id": "Фото 2",
        "filename": "m1.jpg",
        "expected_gender": "Male",
        "path": "/static/dataset/m1.jpg"
    },
    {
        "id": "Фото 3",
        "filename": "f2.jpg",
        "expected_gender": "Female",
        "path": "/static/dataset/f2.jpg"
    },
    {
        "id": "Фото 4",
        "filename": "m2.jpg",
        "expected_gender": "Male",
        "path": "/static/dataset/m2.jpg"
    },
    {
        "id": "Фото 5",
        "filename": "f3.jpg",
        "expected_gender": "Female",
        "path": "/static/dataset/f3.jpg"
    },
    {
        "id": "Фото 6",
        "filename": "m3.jpg",
        "expected_gender": "Male",
        "path": "/static/dataset/f3.jpg"
    }
]


# Load model (with demo fallback)
def load_model_safe():
    try:
        model = tf.keras.models.load_model("model/model.keras")
        print(f"✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"⚠️ Model loading failed: {e}")
        print("🎭 Running in DEMO MODE")
        return None

try:
    model = load_model_safe()
    DEMO_MODE = model is None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    DEMO_MODE = True


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "dataset": PREDEFINED_DATASET,
        "demo_mode": DEMO_MODE
    })

@app.get("/dataset")
async def get_dataset():
    """Return the predefined dataset for frontend"""
    return {"dataset": PREDEFINED_DATASET, "demo_mode": DEMO_MODE}


@app.post("/predict")
async def predict_fish_gender(files: List[UploadFile] = File(...)):
    """Handle uploaded files"""
    if not files:
        return JSONResponse(content={"error": "No files provided"}, status_code=400)

    results = []

    for file in files:
        try:
            image_data = await file.read()
            result = await process_single_image(image_data, file.filename)
            results.append(result)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return JSONResponse(content={
        "results": results,
        "demo_mode": DEMO_MODE
    })

@app.post("/predict-selected")
async def predict_selected_images(selected_ids: List[str]):
    """Handle predefined dataset selection"""
    if not selected_ids:
        return JSONResponse(content={"error": "No images selected"}, status_code=400)

    results = []

    for image_id in selected_ids:
        # Find the image in predefined dataset
        dataset_item = next((item for item in PREDEFINED_DATASET if item["id"] == image_id), None)

        if not dataset_item:
            results.append({
                "filename": image_id,
                "error": "Image not found in dataset"
            })
            continue

        try:
            # Load image from dataset
            image_path = f"static/dataset/{dataset_item['filename']}"

            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    image_data = f.read()

                result = await process_single_image(
                    image_data,
                    dataset_item['filename'],
                    dataset_item['path'],
                    dataset_item
                )
                results.append(result)
            else:
                # Create placeholder result for demo
                result = create_demo_result(dataset_item)
                results.append(result)

        except Exception as e:
            results.append({
                "filename": dataset_item['filename'],
                "error": str(e)
            })

    return JSONResponse(content={
        "results": results,
        "demo_mode": DEMO_MODE
    })

async def process_single_image(image_data: bytes, filename: str, image_path: str = None, dataset_item: dict = None):
    """Process a single image and return prediction result"""

    if DEMO_MODE:
        # Demo predictions with some logic based on expected gender
        await asyncio.sleep(0.3)  # Simulate processing time

        if dataset_item and 'expected_gender' in dataset_item:
            # Bias prediction towards expected gender for more realistic demo
            if dataset_item['expected_gender'] == 'Male':
                prediction = np.random.uniform(0.6, 0.9)  # Higher values = Male
            else:
                prediction = np.random.uniform(0.1, 0.4)  # Lower values = Female
        else:
            prediction = np.random.uniform(0.1, 0.9)
    else:
        # Real model prediction
        image = preprocess_image(image_data, IMG_SIZE_X, IMG_SIZE_Y)
        prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)[0][0]

    # Convert to gender and confidence
    if prediction > CONFIDENCE_THRESHOLD:
        gender = "Male"
        confidence = float(prediction * 100)
    else:
        gender = "Female"
        confidence = float((1 - prediction) * 100)

    # Save uploaded image for display (if not from predefined dataset)
    if image_path is None:
        image_id = str(uuid.uuid4())
        image_path = f"/static/uploads/{image_id}.jpg"
        os.makedirs("static/uploads", exist_ok=True)

        with open(f"static/uploads/{image_id}.jpg", "wb") as f:
            f.write(image_data)

    result = {
        "filename": filename,
        "gender": gender,
        "confidence": round(confidence, 2),
        "raw_prediction": float(prediction),
        "image_path": image_path,
        "demo_mode": DEMO_MODE
    }

    # Add dataset info if available
    if dataset_item:
        result.update({
            "description": dataset_item.get('description', ''),
            "expected_gender": dataset_item.get('expected_gender', ''),
            "is_predefined": True
        })

    return result

def create_demo_result(dataset_item: dict):
    """Create a demo result when actual image file doesn't exist"""
    # Simulate prediction based on expected gender
    if dataset_item['expected_gender'] == 'Male':
        prediction = np.random.uniform(0.65, 0.85)
        gender = "Male"
        confidence = prediction * 100
    else:
        prediction = np.random.uniform(0.15, 0.35)
        gender = "Female"
        confidence = (1 - prediction) * 100

    return {
        "filename": dataset_item['filename'],
        "gender": gender,
        "confidence": round(confidence, 2),
        "raw_prediction": float(prediction),
        "image_path": dataset_item['path'],
        "description": dataset_item['description'],
        "expected_gender": dataset_item['expected_gender'],
        "is_predefined": True,
        "demo_mode": DEMO_MODE
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "dataset_size": len(PREDEFINED_DATASET),
        "version": "1.0.0"
    }
