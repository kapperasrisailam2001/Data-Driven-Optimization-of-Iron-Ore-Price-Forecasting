# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 20:33:35 2025

@author: srisailam kappera
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel(r"C:\Users\srisailam kappera\OneDrive\Desktop\360DigiTMG Project_2\anitha\iron_Data_set.xlsx")

# Convert 'Date' to datetime and sort data in ascending order
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date").reset_index(drop=True)

# Check for missing values
missing_values = df.isnull().sum()

# Display cleaned dataset info
df.info(), df.head(), missing_values


# Plot the time series data
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Price"], label="Iron Price", color="blue")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Iron Price Over Time")
plt.legend()
plt.grid()
plt.show()




# Perform seasonal decomposition
decomposition = seasonal_decompose(df["Price"], model="additive", period=365)

# ADF Test for stationarity
adf_test = adfuller(df["Price"])
adf_result = {"ADF Statistic": adf_test[0], "p-value": adf_test[1]}

# Plot decomposition results
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Trend
axes[0].plot(df["Date"], decomposition.trend, label="Trend", color="red")
axes[0].legend()
axes[0].grid()

# Seasonality
axes[1].plot(df["Date"], decomposition.seasonal, label="Seasonality", color="green")
axes[1].legend()
axes[1].grid()

# Residuals
axes[2].plot(df["Date"], decomposition.resid, label="Residuals", color="purple")
axes[2].legend()
axes[2].grid()

plt.show()

# Display ADF test results
adf_result


""" Trend: Shows an overall increase in iron prices.
Seasonality: Some repeating patterns exist, indicating seasonality.
Residuals: Fluctuate but appear mostly stable.
ADF Test Result:
ADF Statistic = -1.79 (closer to 0 means non-stationary).
p-value = 0.38 (greater than 0.05 → fails to reject the null hypothesis).
Conclusion: The data is non-stationary, meaning we need differencing or transformations for models like ARIMA/SARIMA.

"""
# Apply first-order differencing to make the data stationary
df["Price_Diff"] = df["Price"].diff()

# Perform ADF test again
adf_test_diff = adfuller(df["Price_Diff"].dropna())
adf_result_diff = {"ADF Statistic": adf_test_diff[0], "p-value": adf_test_diff[1]}

# Plot the differenced series
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Price_Diff"], label="Differenced Price", color="blue")
plt.xlabel("Date")
plt.ylabel("Differenced Price (USD)")
plt.title("Iron Price After Differencing")
plt.legend()
plt.grid()
plt.show()

# Display new ADF test results
adf_result_diff



# Split data into training and testing sets (80% train, 20% test)
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

test.to_csv("test_iron_ore_data.csv")
# Train ARIMA model (p=1, d=1, q=1 chosen based on stationarity and differencing)
arima_model = ARIMA(train["Price"], order=(1, 1, 1))
arima_result = arima_model.fit()

# Forecast on test data
test["ARIMA_Pred"] = arima_result.forecast(steps=len(test))

# Evaluate model
mae_arima = mean_absolute_error(test["Price"], test["ARIMA_Pred"])

# Plot actual vs predicted values
plt.figure(figsize=(12, 5))
plt.plot(train["Date"], train["Price"], label="Train Data", color="blue")
plt.plot(test["Date"], test["Price"], label="Actual Price", color="green")
plt.plot(test["Date"], test["ARIMA_Pred"], label="ARIMA Prediction", color="red", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("ARIMA Model - Actual vs Predicted")
plt.legend()
plt.grid()
plt.show()

# Display ARIMA MAE
mae_arima

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Train SARIMA model (p=1, d=1, q=1, seasonal P=1, D=1, Q=1, seasonality=12)
sarima_model = SARIMAX(train["Price"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Forecast on test data
test["SARIMA_Pred"] = sarima_result.forecast(steps=len(test))

# Evaluate model
mae_sarima = mean_absolute_error(test["Price"], test["SARIMA_Pred"])

# Plot actual vs predicted values
plt.figure(figsize=(12, 5))
plt.plot(train["Date"], train["Price"], label="Train Data", color="blue")
plt.plot(test["Date"], test["Price"], label="Actual Price", color="green")
plt.plot(test["Date"], test["SARIMA_Pred"], label="SARIMA Prediction", color="red", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("SARIMA Model - Actual vs Predicted")
plt.legend()
plt.grid()
plt.show()

# Display SARIMA MAE
mae_sarima




# Prepare data for Prophet (rename columns as required by Prophet)
prophet_df = df.rename(columns={"Date": "ds", "Price": "y"})

# Train Prophet model
prophet_model = Prophet()
prophet_model.fit(prophet_df)

# Make future predictions
future_dates = prophet_model.make_future_dataframe(periods=len(test), freq="D")
prophet_forecast = prophet_model.predict(future_dates)

# Extract only forecasted values for test period
test["Prophet_Pred"] = prophet_forecast.set_index("ds").loc[test["Date"], "yhat"].values

# Evaluate model
mae_prophet = mean_absolute_error(test["Price"], test["Prophet_Pred"])

# Plot actual vs predicted values
plt.figure(figsize=(12, 5))
plt.plot(train["Date"], train["Price"], label="Train Data", color="blue")
plt.plot(test["Date"], test["Price"], label="Actual Price", color="green")
plt.plot(test["Date"], test["Prophet_Pred"], label="Prophet Prediction", color="red", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Prophet Model - Actual vs Predicted")
plt.legend()
plt.grid()
plt.show()

# Display Prophet MAE
mae_prophet



# Scale the price data for LSTM & GRU
scaler = MinMaxScaler()
df["Scaled_Price"] = scaler.fit_transform(df["Price"].values.reshape(-1, 1))

# Prepare data for time series modeling
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define sequence length
seq_length = 30

# Split scaled data into train and test sets
train_scaled, test_scaled = df["Scaled_Price"][:train_size], df["Scaled_Price"][train_size:]

# Create sequences
X_train, y_train = create_sequences(train_scaled.values, seq_length)
X_test, y_test = create_sequences(test_scaled.values, seq_length)

# Reshape for LSTM/GRU input (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

# Compile LSTM model
lstm_model.compile(optimizer="adam", loss="mse")

# Train LSTM model
lstm_model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# Predict using LSTM model
lstm_pred_scaled = lstm_model.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled)

# Calculate LSTM MAE
mae_lstm = mean_absolute_error(test["Price"].iloc[seq_length:], lstm_pred.flatten())

# Build GRU model
gru_model = Sequential([
    GRU(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    GRU(50),
    Dropout(0.2),
    Dense(1)
])

# Compile GRU model
gru_model.compile(optimizer="adam", loss="mse")

# Train GRU model
gru_model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# Predict using GRU model
gru_pred_scaled = gru_model.predict(X_test)
gru_pred = scaler.inverse_transform(gru_pred_scaled)

# Calculate GRU MAE
mae_gru = mean_absolute_error(test["Price"].iloc[seq_length:], gru_pred.flatten())

# Return MAE values
mae_lstm, mae_gru

# Plot LSTM Predictions
plt.figure(figsize=(12, 5))
plt.plot(df["Date"].iloc[train_size+seq_length:], df["Price"].iloc[train_size+seq_length:], label="Actual Price", color="green")
plt.plot(df["Date"].iloc[train_size+seq_length:], lstm_pred.flatten(), label="LSTM Prediction", color="red", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("LSTM Model - Actual vs Predicted")
plt.legend()
plt.grid()
plt.show()

# Plot GRU Predictions
plt.figure(figsize=(12, 5))
plt.plot(df["Date"].iloc[train_size+seq_length:], df["Price"].iloc[train_size+seq_length:], label="Actual Price", color="green")
plt.plot(df["Date"].iloc[train_size+seq_length:], gru_pred.flatten(), label="GRU Prediction", color="blue", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("GRU Model - Actual vs Predicted")
plt.legend()
plt.grid()
plt.show()

import joblib
lstm_model.save("lstm_model.h5")  # If LSTM is best
# gru_model.save("gru_model.h5")  # Uncomment if GRU is better

# Save the scaler
joblib.dump(scaler, "scaler.pkl")



# MAE Values
mae_values = [mae_arima, mae_sarima, mae_prophet, mae_lstm, mae_gru]



# Model names
models = ["ARIMA", "SARIMA", "Prophet", "LSTM", "GRU"]

# Create a bar chart
plt.figure(figsize=(8, 5))
plt.bar(models, mae_values, color=["blue", "orange", "green", "red", "purple"])

# Add labels and title
plt.xlabel("Model")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Model Performance Comparison (Lower MAE is Better)")

# Display values on top of bars
for i, v in enumerate(mae_values):
    plt.text(i, v + 0.5, f"{v:.2f}", ha='center', fontsize=12)

# Show the plot
plt.ylim(0, max(mae_values) + 5)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()















