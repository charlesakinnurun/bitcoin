# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# %% [markdown]
# Set a style for the plots for better visualization

# %%
plt.style.use('ggplot')

# %% [markdown]
# Data Loading

# %%
try:
    df = pd.read_csv("coin_Bitcoin.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: 'coin_Bitcoin.csv' was not found. Please make sure the file is in the same directory")
    exit()
df

# %% [markdown]
# Data Preprocessing

# %%
# Rename the columns for clarity and consistency
df.rename(columns={
    "SNo":"serial_number",
    "Name":"name",
    "Symbol":"symbol",
    "Date":"date",
    "High":"high",
    "Low":"low",
    "Open":"open",
    "Close":"close",
    "Volume":"volume",
    "Marketcap":"marketcap"
},inplace=True)

# Check for missing values
df_missing = df.isnull().sum()
print("Missing Values")
print(df_missing)

# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("Duplicated Rows")
print(df_duplicated)

# Drop any rows with missing values
df.dropna(inplace=True)

# Convert the "date" column to a numerical format for plotting and potential modelling
df["date"] = pd.to_datetime(df["date"])
df["date_ordinal"] = df["date"].apply(lambda date : date.toordinal())

# %% [markdown]
# Feature Engineering

# %%
# Select the features (X) and target variable (y)
# We'll use the "high","low","open" and "volume" to predict the "close" price

features = ["high","low","open","volume"]
X = df[features]
y = df["close"]

# %% [markdown]
# Visualization before training

# %%
# Create a visualization of the raw data to show the overall trend of Bitcoin's closing price
plt.figure(figsize=(14,7))
plt.plot(df["date"],y,label="Bitcoin Closing Price",color="green")
plt.title("Bitcoin Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Create a scatter plot to visualize the relationshp between "high" price and the "close" price
plt.figure(figsize=(10,6))
sns.scatterplot(x="high",y="close",data=df)
plt.title("High Price vs Close Price")
plt.xlabel("High Price (USD)")
plt.ylabel("Close Price (USD)")
plt.grid(True)
plt.show()

# %%
# Create a scatter plot to visualize the relationshp between "low" price and the "close" price
plt.figure(figsize=(10,6))
sns.scatterplot(x="low",y="close",data=df,color="b")
plt.title("Low Price vs Close Price")
plt.xlabel("Low Price (USD)")
plt.ylabel("Close Price (USD)")
plt.grid(True)
plt.show()

# %%
# Create a scatter plot to visualize the relationshp between "open" price and the "close" price
plt.figure(figsize=(10,6))
sns.scatterplot(x="open",y="close",data=df,color="y")
plt.title("Open Price vs Close Price")
plt.xlabel("Open Price (USD)")
plt.ylabel("Close Price (USD)")
plt.grid(True)
plt.show()

# %%
# Create a scatter plot to visualize the relationshp between "volume" and the "close" price
plt.figure(figsize=(10,6))
sns.scatterplot(x="volume",y="close",data=df,color="black")
plt.title("Volume vs Close Price")
plt.xlabel("Volume (USD)")
plt.ylabel("Close Price (USD)")
plt.grid(True)
plt.show()

# %% [markdown]
# Data Splitting

# %%
# We'll use the 80% of the data for training and 20% for testing to evaluate the model performance.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# %% [markdown]
# Model Training

# %%
# Initialize and train four different regression models

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

# Ridge Regression 
ridge_reg = Ridge()
ridge_reg.fit(X_train,y_train)

# Lasso Regression
lasso_reg = Lasso()
lasso_reg.fit(X_train,y_train)

# ElasticNet Regression
elastic_reg = ElasticNet()
elastic_reg.fit(X_train,y_train)


# Make the predictions on the test set of each model
y_pred_lin = lin_reg.predict(X_test)
y_pred_ridge = ridge_reg.predict(X_test)
y_pred_lasso = lasso_reg.predict(X_test)
y_pred_elastic = elastic_reg.predict(X_test)

# %% [markdown]
# Model Evaluation

# %%
# Linear Regression Metrics
print("-----Linear Regression-----")
print(f"R-squared: {r2_score(y_test,y_pred_lin):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred_lin):.4f}")
print(f"MSE: {mean_squared_error(y_test,y_pred_lin):.4f}")

# Ridge Regression Metrics
print("-----Ridge Regression-----")
print(f"R-squared: {r2_score(y_test,y_pred_ridge):.4f}")
print(f"MAE: {r2_score(y_test,y_pred_ridge):.4f}")
print(f"MSE: {mean_absolute_error(y_test,y_pred_ridge):.4f}")

# Lasso Regression Metrics
print("-----Lasso Regression-----")
print(f"R-squared: {r2_score(y_test,y_pred_lasso):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred_lasso):.4f}")
print(f"MSE: {mean_squared_error(y_test,y_pred_lasso):.4f}")

# ElasticNet Regression Metrics
print("-----Elastic Regression-----")
print(f"R-squared: {r2_score(y_test,y_pred_elastic):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred_elastic):.4f}")
print(f"MSE: {mean_squared_error(y_test,y_pred_elastic):.4f}")

# %% [markdown]
# Determine the best model

# %%
# Determine the best model based on R-squared score (or other metrics)
# In this case, R-squared is a good indicator of overall fit
r2_scores = {
    "Linear": r2_score(y_test,y_pred_lin),
    "Ridge": r2_score(y_test,y_pred_ridge),
    "Lasso": r2_score(y_test,y_pred_lasso),
    "ElasticNet": r2_score(y_test,y_pred_elastic)
}

best_model_name = max(r2_scores,key=r2_scores.get)
print(f"Conclusion: The best performing model is {best_model_name} Regression")


# Select the best model's predictions for the final visualization
if best_model_name == "Linear":
    y_pred_best = y_pred_lin
elif best_model_name == "Ridge":
    y_pred_best = y_pred_ridge
elif best_model_name == "Lasso":
    y_pred_best = y_pred_lasso
else:
    y_pred_best = y_pred_elastic

# %% [markdown]
# Visualization after training

# %%
# Create a visualization to compare the actual values with the predictions from the best model
# This plot shows how closely the model's predictions align with real data
plt.figure(figsize=(14,7))
plt.scatter(range(len(y_test)),y_test,color="blue",label="Actual Prices")
plt.scatter(range(len(y_pred_best)),y_pred_best,color="red",label="Predicted Prices")
plt.title(f"Actual v Predicted Prices ({best_model_name} Regression)")
plt.xlabel("Test Sample Index")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# User Input and Prediction

# %%
print("------Predict Bitcoin Closing Price-----")
print("Enter the following data to predict the closing price:")

try:
    # Prompt the user for the feature values
    high_price = float(input("Enter the High Price:"))
    low_price = float(input("Enter the Low price:"))
    open_price = float(input("Enter the Open Price:"))
    volume = float(input("Enter the Volume:"))

    # Create a new DataFrame with user's input
    # The data must be in the same format as the training data
    new_data = pd.DataFrame([[high_price,low_price,open_price,volume]],columns=features)

    # Use the best-performing model to make a prediction on the new data
    if best_model_name == "Linear":
        predicted_price = lin_reg.predict(new_data)
    elif best_model_name == "Ridge":
        predicted_price = ridge_reg.predict(new_data)
    elif best_model_name == "Lasso":
        predicted_price = lasso_reg.predict(new_data)
    else:
        predicted_price = elastic_reg.predict(new_data)

    # Print the final predicted price
    print(f"Predicted Closing Price: ${predicted_price[0]:.2f}")

except ValueError:
    print("Invalid input. Please enter valid numerical values.")
except Exception as e:
    print(f"An error occurred: {e}")


