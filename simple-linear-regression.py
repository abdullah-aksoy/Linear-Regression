# Sales prediction using linear regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure pandas to display float numbers with two decimal places
pd.set_option('display.float_format', lambda x: "%.2f" % x)

# Importing necessary tools for model creation and evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset
df = pd.read_csv("machine-learning/dataset/advertising.csv")

# Print the first few rows of the dataset to inspect it
print(df.head())

# Print the column names to ensure they are correct
print("Column names in the dataset:", df.columns)

# Check for missing values
if df.isnull().sum().any():
    print("Dataset contains missing values. Please handle them before proceeding.")
else:
    print("No missing values in the dataset.")

# Check the shape of the dataset
print("Shape of the dataset:", df.shape)

# Define the independent variable (X) and the dependent variable (y)
X = df[["TV"]]
y = df["Sales"]  # Ensure the column name matches exactly

# Fit the linear regression model
reg_model = LinearRegression().fit(X, y)

# Intercept (bias term)
print("Intercept:", reg_model.intercept_)

# Coefficient - the weight associated with TV advertising
print("Coefficient:", reg_model.coef_[0])

# Predict sales for a TV advertising budget of 500 units
predicted_sales = reg_model.intercept_ + reg_model.coef_[0] * 500
print("Predicted sales for 500 units of TV advertising:", predicted_sales)

# Summarize the dataset
print(df.describe().T)

#-----------------------------------------------------------

# Visualization of the model
# Note: Using "scatter_kws" customizes the scatter points. If replaced with "scatter=False", only the regression line will be shown.
g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9}, ci=95, color="r")

# Add a title and labels for the plot
g.set_title(f"Model Equation: Sales = {round(reg_model.intercept_, 2)} + TV*{round(reg_model.coef_[0], 2)}")
g.set_ylabel("Number of Sales")
g.set_xlabel("TV Advertising Budget")
plt.xlim(-10, 310)  # Set limits for the x-axis

# Start y-axis from 0
plt.ylim(bottom=0)

# Display the plot
plt.show()

#-----------------------------------------------------------

# Mean Squared Error (MSE)
# Used to evaluate the performance of the linear regression model.
# The model predicts the dependent variable (sales) based on the independent variable (TV spending).

y_pred = reg_model.predict(X)
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error (MSE):", mse)

# Calculate the mean of the dependent variable (sales)
print("Mean of sales:", y.mean())

# Calculate the standard deviation of sales
print("Standard deviation of sales:", y.std())

# Root Mean Squared Error (RMSE) - Square root of the MSE
rmse = np.sqrt(mean_squared_error(y, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y, y_pred)
print("Mean Absolute Error (MAE):", mae)

# R-squared (R²) - Percentage of variance in the dependent variable explained by the independent variable
r_squared = reg_model.score(X, y)
print("R-squared (R²):", r_squared)

# Note:
# We are not conducting coefficient tests. Instead, we focus on machine learning and optimization perspectives.
# While advanced regression models and tree-based regression methods provide solutions, understanding the basics of linear regression is still essential.