# Improved Room Classification Application

This is an improved version of the messy vs clean room classification application with better model architecture and enhanced API features.

## Features

### Improved Model (`training_model.py`)
- **Transfer Learning**: Uses MobileNetV2 pre-trained on ImageNet for better accuracy
- **Advanced Data Augmentation**: Rotation, shifts, zoom, brightness adjustments
- **Regularization**: Dropout layers and L2 regularization to prevent overfitting
- **Batch Normalization**: Faster convergence and better stability
- **Two-Phase Training**: 
  - Phase 1: Train with frozen base model
  - Phase 2: Fine-tune top layers
- **Smart Callbacks**: Early stopping, learning rate reduction, model checkpointing
- **Comprehensive Metrics**: Accuracy, Precision, Recall

### Enhanced API (`app.py`)
- **Multiple Endpoints**:
  - `GET /` - API information
  - `GET /health` - Health check
  - `GET /model/info` - Model information
  - `POST /predict` - Single image prediction
  - `POST /predict/batch` - Batch prediction (up to 10 images)
- **Confidence Scores**: Returns probability scores for both classes
- **File Validation**: Checks file type and size
- **CORS Support**: Ready for frontend integration
- **Error Handling**: Comprehensive error messages
- **Auto-Detection**: Automatically uses improved model if available

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Improved Model

```bash
python training_model.py
```

This will:
- Extract the dataset from `messy-vs-clean-room.zip`
- Train the improved model with transfer learning
- Save the model to `trained_model_improved/`

### 2. Run the API Server

```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 3. API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Model Information
```bash
curl http://localhost:8000/model/info
```

#### Single Image Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/your/image.jpg"
```

#### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### 4. Interactive Documentation

Visit `http://localhost:8000/docs` for Swagger UI documentation.

## Example Response

### Single Prediction
```json
{
  "predicted_class": "messy",
  "label_indonesian": "Kamar berantakan",
  "confidence": 94.52,
  "probability": {
    "messy": 94.52,
    "clean": 5.48
  },
  "filename": "room.jpg",
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

### Batch Prediction
```json
{
  "successful_predictions": 2,
  "errors": 0,
  "results": [
    {
      "predicted_class": "clean",
      "label_indonesian": "Kamar rapi",
      "confidence": 87.34,
      "probability": {
        "messy": 12.66,
        "clean": 87.34
      },
      "filename": "room1.jpg"
    },
    {
      "predicted_class": "messy",
      "label_indonesian": "Kamar berantakan",
      "confidence": 92.15,
      "probability": {
        "messy": 92.15,
        "clean": 7.85
      },
      "filename": "room2.jpg"
    }
  ],
  "errors_detail": null,
  "timestamp": "2024-01-15T10:31:22.654321"
}
```

## Model Comparison

| Feature | Original | Improved |
|---------|----------|----------|
| Architecture | Custom CNN | MobileNetV2 (Transfer Learning) |
| Input Size | 150x150 | 224x224 |
| Parameters | ~20M | ~3M (trainable: ~500K) |
| Data Augmentation | Basic | Advanced |
| Regularization | None | Dropout + L2 |
| Training Strategy | Single phase | Two-phase (freeze + fine-tune) |
| Metrics | Accuracy only | Accuracy, Precision, Recall |
| Expected Accuracy | ~75-85% | ~90-95% |

## Project Structure

```
/workspace
├── training_model.py      # Improved model training script
├── app.py                 # Enhanced FastAPI application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── messy-vs-clean-room.zip  # Dataset
├── trained_model/         # Original trained model
└── trained_model_improved/ # New improved model (after training)
```

## License

MIT License
