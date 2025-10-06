from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, Union
import pandas as pd
import traceback
import logging

from .data.preprocessor import DataPreprocessor
from .model.predictor import ModelPredictor
from .logger import setup_logging

# Setup logging
logger: logging.Logger = setup_logging()

# Initialize preprocessor and predictor
try:
    preprocessor: DataPreprocessor = DataPreprocessor()
    predictor: ModelPredictor = ModelPredictor()
except Exception as e:
    logger.error(f"Initialization error: {e}")
    logger.error(traceback.format_exc())
    raise

class HouseInputLite(BaseModel):
    """
    Lightweight model for house price prediction with minimal features.
    
    Attributes:
        price (Optional[float]): Optional price of the house.
        bedrooms (float): Number of bedrooms.
        bathrooms (float): Number of bathrooms.
        sqft_living (float): Square footage of living area.
        sqft_lot (float): Square footage of the lot.
        floors (float): Number of floors.
        sqft_above (float): Square footage above ground.
        sqft_basement (float): Square footage of basement.
        zipcode (float): Zipcode of the property.
    """
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
    """
    Comprehensive model for house price prediction with full feature set.
    
    Attributes:
        date (Optional[str]): Optional date of listing.
        price (Optional[float]): Optional price of the house.
        bedrooms (float): Number of bedrooms.
        bathrooms (float): Number of bathrooms.
        sqft_living (float): Square footage of living area.
        sqft_lot (float): Square footage of the lot.
        floors (float): Number of floors.
        waterfront (int): Waterfront property indicator.
        view (int): View rating.
        condition (int): Property condition rating.
        grade (int): Construction and design quality rating.
        sqft_above (float): Square footage above ground.
        sqft_basement (float): Square footage of basement.
        yr_built (int): Year the house was built.
        yr_renovated (int): Year of last renovation.
        zipcode (int): Zipcode of the property.
        lat (float): Latitude coordinate.
        long (float): Longitude coordinate.
        sqft_living15 (float): Living room area in 2015.
        sqft_lot15 (float): Lot area in 2015.
    """
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
async def predict_house_price(house: HouseInput) -> Dict[str, Union[float, Dict[str, Any], Dict[str, float]]]:
    """
    Predict house price using full feature set.

    Args:
        house (HouseInput): Comprehensive house details for prediction.

    Returns:
        Dict containing prediction, metadata, and input features.

    Raises:
        HTTPException: For preprocessing or prediction errors.
    """
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
async def predict_house_price_lite(house: HouseInputLite) -> Dict[str, Union[float, Dict[str, Any], Dict[str, float]]]:
    """
    Predict house price using a minimal set of features.

    Args:
        house (HouseInputLite): Minimal house details for prediction.

    Returns:
        Dict containing prediction, metadata, and input features.

    Raises:
        HTTPException: For preprocessing or prediction errors.
    """
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
