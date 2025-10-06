import requests
import pandas as pd
import json

def showcase_prediction_api():
    # Read the future unseen examples
    df = pd.read_csv('data/future_unseen_examples.csv')
    
    # Select first 5 rows for demonstration
    demo_samples = df.head(5)
    
    # Endpoint URL
    url = 'http://localhost:8000/predict'
    
    print("House Price Prediction API Showcase")
    print("===================================")
    
    # Demonstrate prediction for each sample
    for index, row in demo_samples.iterrows():
        # Convert row to dictionary, removing any NaN values
        input_data = row.dropna().to_dict()
        
        print(f"\nSample {index + 1} Input:")
        print(json.dumps(input_data, indent=2))
        
        # Send POST request
        response = requests.post(url, json=input_data)
        
        # Print prediction
        if response.status_code == 200:
            prediction = response.json()['prediction']
            print(f"Predicted House Price: ${prediction:,.2f}")
        else:
            print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    showcase_prediction_api()
