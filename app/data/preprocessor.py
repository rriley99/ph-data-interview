import pandas as pd
import json

class DataPreprocessor:
    def __init__(self, demographics_path='data/zipcode_demographics.csv'):
        self.demographics = pd.read_csv(demographics_path)
        
        with open('model/model_features.json', 'r') as f:
            self.model_features = json.load(f)
    
    def preprocess_data(self, input_data):
        """
        Merge input house data with demographic data based on zipcode
        Ensure features are in the correct order for model prediction
        """
        # Merge input data with demographics
        merged_data = input_data.merge(
            self.demographics, 
            left_on='zipcode', 
            right_on='zipcode', 
            how='left'
        )
        
        # Select and order features exactly as the model expects
        processed_data = merged_data[self.model_features]
        
        return processed_data
