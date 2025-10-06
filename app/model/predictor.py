import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger('sound_realty_api')

class ModelPredictor:
    def __init__(self, model_path='model/model.pkl'):
        try:
            self.model = joblib.load(model_path)
            self.model_version = '1.0.0'
            self.loaded_at = datetime.now()
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, processed_data):
        """
        Make predictions using the loaded model
        """
        try:
            # Log input data details for debugging
            logger.info(f"Input data shape: {processed_data.shape}")
            logger.info(f"Input data columns: {processed_data.columns.tolist()}")
            
            # Ensure input is a DataFrame
            if not isinstance(processed_data, pd.DataFrame):
                processed_data = pd.DataFrame(processed_data)
            
            # Make prediction
            predictions = self.model.predict(processed_data)
            
            # Log prediction details
            logger.info(f"Prediction made. Number of predictions: {len(predictions)}")
            
            return predictions
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def get_metadata(self):
        """
        Generate metadata about the model and prediction
        """
        return {
            'model_version': self.model_version,
            'model_type': type(self.model).__name__,
            'model_loaded_at': str(self.loaded_at),
            'prediction_timestamp': str(datetime.now())
        }
