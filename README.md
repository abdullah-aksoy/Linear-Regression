# Sales Prediction Using Linear Regression

This project demonstrates how to use linear regression to predict sales based on TV advertising budgets. The dataset used contains information on advertising budgets for TV, radio, and newspaper, and the corresponding sales figures.

![myplot](https://github.com/user-attachments/assets/5babcfb3-af15-4454-998a-5ce13ac6cc28)

## Features

- Load and inspect the dataset
- Fit a linear regression model
- Visualize the regression model
- Evaluate the model using various metrics

## Requirements

- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/abdullah-aksoy/Simple-Linear-Regression
    cd Simple-Linear-Regression
    ```

2. Install the required libraries:
    ```sh
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```

## Usage

1. Ensure the dataset `advertising.csv` is located in the `machine-learning/dataset/` directory.

2. Run the script:
    ```sh
    python sales_prediction.py
    ```

## Code Explanation

The script performs the following steps:

1. **Import Libraries**: Import necessary libraries for data manipulation, visualization, and model creation.
    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    ```

2. **Load Dataset**: Load the dataset and inspect the first few rows and column names.

3. **Check for Missing Values**: Ensure there are no missing values in the dataset.

4. **Define Variables**: Define the independent variable (TV advertising budget) and the dependent variable (sales).

5. **Fit Linear Regression Model**: Fit the linear regression model and display the intercept and coefficient.

6. **Predict Sales**: Predict sales for a given TV advertising budget.

7. **Summarize Dataset**: Display summary statistics of the dataset.

8. **Visualize Model**: Visualize the regression model using seaborn.

9. **Evaluate Model**: Evaluate the model using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²).

