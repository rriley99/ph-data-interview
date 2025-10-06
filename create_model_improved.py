import json
import pathlib
import pickle
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import metrics

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containing two elements: a DataFrame and a Series of the same
        length. The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).
    """
    data = pd.read_csv(sales_path,
                       usecols=sales_column_selection,
                       dtype={'zipcode': str})
    demographics = pd.read_csv(demographics_path,
                               dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data

    return x, y


def evaluate_model(model, x, y, n_splits: int = 5) -> Dict[str, Any]:
    """
    Perform cross-validation and calculate performance metrics.

    Args:
        model: Scikit-learn pipeline or model
        x: Feature DataFrame
        y: Target Series
        n_splits: Number of cross-validation splits

    Returns:
        Dictionary of performance metrics
    """
    # Perform cross-validation
    cv_results = model_selection.cross_validate(
        model, x, y, 
        cv=n_splits, 
        scoring=['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
    )

    # Calculate metrics
    metrics_dict = {
        'r2_scores': cv_results['test_r2'].tolist(),
        'mae_scores': np.abs(cv_results['test_neg_mean_absolute_error']).tolist(),
        'mse_scores': np.abs(cv_results['test_neg_mean_squared_error']).tolist(),
        'rmse_scores': np.sqrt(np.abs(cv_results['test_neg_mean_squared_error'])).tolist(),
        'mean_r2': np.mean(cv_results['test_r2']),
        'mean_mae': np.mean(np.abs(cv_results['test_neg_mean_absolute_error'])),
        'mean_mse': np.mean(np.abs(cv_results['test_neg_mean_squared_error'])),
        'mean_rmse': np.mean(np.sqrt(np.abs(cv_results['test_neg_mean_squared_error'])))
    }

    return metrics_dict


def main():
    """Load data, train model, evaluate, and export artifacts."""
    # Load data
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

    # Split data into train and final holdout test sets
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2, random_state=42)

    # Create model pipeline
    model = pipeline.make_pipeline(
        preprocessing.RobustScaler(),
        neighbors.KNeighborsRegressor()
    )

    # Fit the model on training data
    model.fit(x_train, y_train)

    # Evaluate model with cross-validation
    cv_metrics = evaluate_model(model, x_train, y_train)

    # Evaluate on holdout test set
    y_pred = model.predict(x_test)
    holdout_metrics = {
        'holdout_r2': metrics.r2_score(y_test, y_pred),
        'holdout_mae': metrics.mean_absolute_error(y_test, y_pred),
        'holdout_mse': metrics.mean_squared_error(y_test, y_pred),
        'holdout_rmse': np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    }

    # Combine metrics
    full_metrics = {**cv_metrics, **holdout_metrics}

    # Create output directory
    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Output model artifacts
    pickle.dump(model, open(output_dir / "model.pkl", 'wb'))
    json.dump(list(x_train.columns), 
              open(output_dir / "model_features.json", 'w'))
    
    # Save performance metrics
    with open(output_dir / "model_performance.json", 'w') as f:
        json.dump(full_metrics, f, indent=4)

    # Print performance summary
    print("Model Performance Summary:")
    for key, value in full_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
