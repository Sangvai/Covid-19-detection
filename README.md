# Covid-19-detection

## Overview
This project uses a Random Forest Regressor to predict COVID-19 deaths based on features such as confirmed cases, recovered cases, and the number of days since the start of the pandemic. The model achieves high accuracy with an R² score of 1.00 and a Mean Absolute Error (MAE) of 24.64.

## Features
- **Data Preprocessing**: Converts dates to numerical values and encodes categorical columns.
- **Model Training**: Uses Random Forest Regressor with hyperparameter tuning via GridSearchCV and RandomizedSearchCV.
- **Evaluation**: Evaluates model performance using R² score and MAE.
- **Feature Importance**: Visualizes the importance of each feature in predictions.
- **Accuracy Check**: Plots actual vs. predicted deaths to validate model accuracy.

## Dataset
The dataset used is `covid_19_clean_complete.csv`, which includes:
- Date
- Confirmed cases
- Deaths
- Recovered cases
- Country/Region
- Latitude and Longitude
- WHO Region

## Model Performance
- **R² Score**: 1.00
- **Mean Absolute Error (MAE)**: 24.64

## Key Visualizations
1. **Feature Importance**: Shows the contribution of each feature to the model's predictions.
2. **Actual vs. Predicted Deaths**: A scatter plot comparing actual deaths to model predictions, with a perfect prediction line for reference.

## Hyperparameter Tuning
The model was optimized using:
- **GridSearchCV**: Tested combinations of `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- **RandomizedSearchCV**: Efficiently explored hyperparameter space with fewer iterations.

## Requirements
- Python 3.11.13
- Libraries:
  - pandas
  - scikit-learn
  - numpy
  - matplotlib

## Usage
1. Clone the repository.
2. Install the required libraries.
3. Run the Jupyter notebook `covid19-model-regression.ipynb` to train and evaluate the model.

## Future Improvements
- Incorporate additional features like vaccination rates or government response measures.
- Experiment with other regression models (e.g., XGBoost, SVM) for comparison.
- Deploy the model as a web application for real-time predictions.

## Author
Sanika Govardhan Sangvai

## License
This project is open-source. Feel free to use and modify the code as needed.
