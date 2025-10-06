# Sound Realty House Price Prediction API

## Project Overview
This project provides a FastAPI-based service for predicting house prices using a pre-trained machine learning model, enriched with demographic data.

## Features
- Full prediction endpoint (`/predict`): Requires all house features
- Lite prediction endpoint (`/predict_lite`): Requires minimal features
- Automatic demographic data enrichment
- Structured logging
- Metadata generation for each prediction

## Prerequisites
- Python 3.9+
- Docker (optional)

## Installation

1. Clone the repository
2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

### Local Development
```bash
uvicorn app.main:app --reload
```

### Docker
```bash
docker build -t sound-realty-api .
docker run -p 8000:8000 sound-realty-api
```

## Testing the API

### Manual Testing
Use the included test script:
```bash
python showcase.py
```

### API Documentation
When the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure
- `app/`: Main application code
  - `data/preprocessor.py`: Data preprocessing logic
  - `model/predictor.py`: Model loading and prediction
  - `main.py`: FastAPI application
  - `logger.py`: Logging configuration
- `model/`: Trained model artifacts
- `data/`: Input data files
- `test_api.py`: API testing script

## Endpoints
- `/predict`: Full feature prediction
- `/predict_lite`: Minimal feature prediction

## Logging
Logs are generated in the `logs/` directory with timestamps.
