from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import traceback

from .data.preprocessor import DataPreprocessor
from .model.predictor import ModelPredictor
from .logger import setup_logging

# Setup logging
logger = setup_logging()

# Initialize preprocessor and predictor
try:
    preprocessor = DataPreprocessor()
    predictor = ModelPredictor()
except Exception as e:
    logger.error(f"Initialization error: {e}")
    logger.error(traceback.format_exc())
    raise

class HouseInputLite(BaseModel):
    price: Optional[float] = None
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    sqft_above: float
    sqft_basement: float
    zipcode: float

# Pydantic model for full input validation
class HouseInput(BaseModel):
    date: Optional[str] = None
    price: Optional[float] = None
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: float
    sqft_basement: float
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float

# Create FastAPI app
app = FastAPI(
    title="Sound Realty House Price Prediction API",
    description="API for predicting house prices with demographic enrichment"
)

@app.post("/predict")
async def predict_house_price(house: HouseInput):
    try:
        # Remove optional fields if present
        input_dict = house.model_dump()
        input_dict.pop('date', None)
        input_dict.pop('price', None)
        
        # Log input data for debugging
        logger.info(f"Received prediction request for house: {input_dict}")
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Preprocess data
        try:
            processed_data = preprocessor.preprocess_data(input_df)
        except Exception as preprocess_error:
            logger.error(f"Preprocessing error: {preprocess_error}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=400, detail=f"Data preprocessing failed: {str(preprocess_error)}")
        
        # Make prediction
        try:
            prediction = predictor.predict(processed_data)[0]
        except Exception as predict_error:
            logger.error(f"Prediction error: {predict_error}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(predict_error)}")
        
        # Get metadata
        metadata = predictor.get_metadata()
        
        # Log prediction details
        logger.info(f"Prediction made: ${prediction:.2f} for zipcode {input_dict['zipcode']}")
        
        return {
            "prediction": float(prediction),
            "metadata": metadata,
            "input_features": input_dict
        }
    
    except HTTPException:
        # Re-raise HTTPException to preserve status code and detail
        raise
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Unexpected error in prediction: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="An unexpected error occurred during prediction")

# Lite endpoint with minimal features
@app.post("/predict_lite")
async def predict_house_price_lite(house: HouseInputLite):
    try:
        # Remove optional fields if present
        input_dict = house.dict()
        input_dict.pop('date', None)
        input_dict.pop('price', None)
        
        # Create DataFrame with input features
        input_df = pd.DataFrame([input_dict])
        
        # Preprocess data
        processed_data = preprocessor.preprocess_data(input_df)
        
        # Make prediction
        prediction = predictor.predict(processed_data)[0]
        
        # Get metadata
        metadata = predictor.get_metadata()
        
        # Log prediction details
        logger.info(f"Lite prediction made: ${prediction:.2f} for zipcode {input_dict['zipcode']}")
        
        return {
            "prediction": float(prediction),
            "metadata": metadata,
            "input_features": input_dict
        }
    
    except Exception as e:
        logger.error(f"Lite prediction error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
