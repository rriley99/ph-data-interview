import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Union, Dict, Any, List

logger: logging.Logger = logging.getLogger('sound_realty_api')

class ModelPredictor:
    """
    A class responsible for loading a pre-trained machine learning model 
    and making predictions on processed house data.

    This class handles model loading, prediction, and metadata generation 
    for a house price prediction model.

    Attributes:
        model: The loaded machine learning model.
        model_version (str): Version of the loaded model.
        loaded_at (datetime): Timestamp when the model was loaded.
    """

    def __init__(self, model_path: str = 'model/model.pkl'):
        """
        Initialize the ModelPredictor by loading a pre-trained model.

        Args:
            model_path (str, optional): Path to the pickled model file. 
                Defaults to 'model/model.pkl'.

        Raises:
            Exception: If there's an error loading the model.
        """
        try:
            self.model = joblib.load(model_path)
            self.model_version: str = '1.0.0'
            self.loaded_at: datetime = datetime.now()
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, processed_data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the loaded machine learning model.

        This method takes preprocessed input data, ensures it's in the correct format, 
        and uses the loaded model to generate predictions.

        Args:
            processed_data (Union[pd.DataFrame, np.ndarray]): Preprocessed input data 
                containing house features in the order expected by the model.

        Returns:
            np.ndarray: Predicted house prices.

        Raises:
            Exception: If there's an error during prediction.
        """
        try:
            # Log input data details for debugging
            logger.info(f"Input data shape: {processed_data.shape}")
            logger.info(f"Input data columns: {processed_data.columns.tolist() if isinstance(processed_data, pd.DataFrame) else 'N/A'}")
            
            # Ensure input is a DataFrame
            if not isinstance(processed_data, pd.DataFrame):
                processed_data = pd.DataFrame(processed_data)
            
            # Make prediction
            predictions: np.ndarray = self.model.predict(processed_data)
            
            # Log prediction details
            logger.info(f"Prediction made. Number of predictions: {len(predictions)}")
            
            return predictions
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def get_metadata(self) -> Dict[str, str]:
        """
        Generate metadata about the model and current prediction context.

        Returns:
            Dict[str, str]: A dictionary containing model and prediction metadata.
                - 'model_version': Version of the loaded model
                - 'model_type': Type/name of the machine learning model
                - 'model_loaded_at': Timestamp when the model was loaded
                - 'prediction_timestamp': Current timestamp
        """
        return {
            'model_version': self.model_version,
            'model_type': type(self.model).__name__,
            'model_loaded_at': str(self.loaded_at),
            'prediction_timestamp': str(datetime.now())
        }
