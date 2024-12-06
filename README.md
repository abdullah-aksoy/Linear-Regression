# Sales Prediction Using Linear Regression

This project demonstrates how to use linear regression to predict sales based on TV advertising budgets. The dataset used contains information on advertising budgets for TV, radio, and newspaper, and the corresponding sales figures.

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
    ```python
    df = pd.read_csv("machine-learning/dataset/advertising.csv")
    print(df.head())
    print("Column names in the dataset:", df.columns)
    ```

3. **Check for Missing Values**: Ensure there are no missing values in the dataset.
    ```python
    if df.isnull().sum().any():
        print("Dataset contains missing values. Please handle them before proceeding.")
    else:
        print("No missing values in the dataset.")
    ```

4. **Define Variables**: Define the independent variable (TV advertising budget) and the dependent variable (sales).
    ```python
    X = df[["TV"]]
    y = df["Sales"]
    ```

5. **Fit Linear Regression Model**: Fit the linear regression model and display the intercept and coefficient.
    ```python
    reg_model = LinearRegression().fit(X, y)
    print("Intercept:", reg_model.intercept_)
    print("Coefficient:", reg_model.coef_[0])
    ```

6. **Predict Sales**: Predict sales for a given TV advertising budget.
    ```python
    predicted_sales = reg_model.intercept_ + reg_model.coef_[0] * 500
    print("Predicted sales for 500 units of TV advertising:", predicted_sales)
    ```

7. **Summarize Dataset**: Display summary statistics of the dataset.
    ```python
    print(df.describe().T)
    ```

8. **Visualize Model**: Visualize the regression model using seaborn.
    ```python
    g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9}, ci=95, color="r")
    g.set_title(f"Model Equation: Sales = {round(reg_model.intercept_, 2)} + TV*{round(reg_model.coef_[0], 2)}")
    g.set_ylabel("Number of Sales")
    g.set_xlabel("TV Advertising Budget")
    plt.xlim(-10, 310)
    plt.ylim(bottom=0)
    plt.show()
    ```

9. **Evaluate Model**: Evaluate the model using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²).
    ```python
    y_pred = reg_model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print("Mean Squared Error (MSE):", mse)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print("Root Mean Squared Error (RMSE):", rmse)
    mae = mean_absolute_error(y, y_pred)
    print("Mean Absolute Error (MAE):", mae)
    r_squared = reg_model.score(X, y)
    print("R-squared (R²):", r_squared)
    ```
