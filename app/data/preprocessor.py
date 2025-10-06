import pandas as pd
import json
from typing import List, Union

class DataPreprocessor:
    """
    A class responsible for preprocessing house data by merging with demographic information.

    This preprocessor combines input house data with demographic data based on zipcode
    and ensures that the features are prepared in the exact order expected by the prediction model.

    Attributes:
        demographics (pd.DataFrame): DataFrame containing demographic information by zipcode.
        model_features (List[str]): List of feature names in the order expected by the model.
    """

    def __init__(self, demographics_path: str = 'data/zipcode_demographics.csv'):
        """
        Initialize the DataPreprocessor with demographic data and model features.

        Args:
            demographics_path (str, optional): Path to the CSV file containing demographic data. 
                Defaults to 'data/zipcode_demographics.csv'.
        """
        self.demographics: pd.DataFrame = pd.read_csv(demographics_path)
        
        with open('model/model_features.json', 'r') as f:
            self.model_features: List[str] = json.load(f)
    
    def preprocess_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input house data by merging with demographic information.

        This method performs two key operations:
        1. Merges input house data with demographic data based on zipcode
        2. Selects and orders features exactly as the model expects

        Args:
            input_data (pd.DataFrame): Input DataFrame containing house features.

        Returns:
            pd.DataFrame: Processed DataFrame with merged demographic data and features 
                          in the order expected by the prediction model.
        """
        # Merge input data with demographics
        merged_data: pd.DataFrame = input_data.merge(
            self.demographics, 
            left_on='zipcode', 
            right_on='zipcode', 
            how='left'
        )
        
        # Select and order features exactly as the model expects
        processed_data: pd.DataFrame = merged_data[self.model_features]
        
        return processed_data
