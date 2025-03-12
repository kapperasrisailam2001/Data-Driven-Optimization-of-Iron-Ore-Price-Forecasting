import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load LSTM model with a custom loss function
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

try:
    model = tf.keras.models.load_model("lstm_model.h5", custom_objects={"mse": mse})
    st.success("✅ LSTM Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Function to load and preprocess test data
def load_and_preprocess_test_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Detect the date column
    date_column = [col for col in df.columns if "date" in col.lower()]
    
    if not date_column:
        st.error("❌ No date column found. Ensure the CSV has a valid date column.")
        return None, None, None

    date_column = date_column[0]
    df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, errors="coerce")
    df.dropna(inplace=True)

    # Select only numerical columns for prediction
    numerical_cols = df.select_dtypes(include=["number"]).columns

    if numerical_cols.empty:
        st.error("❌ No numerical columns found for prediction.")
        return None, None, None

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[numerical_cols])

    return df_scaled, scaler, df[date_column]

# Streamlit UI
st.title("🔮 Iron Ore Price Prediction")

uploaded_file = st.file_uploader("📂 Upload your test dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    X_test, scaler, date_series = load_and_preprocess_test_data(uploaded_file)

    if X_test is not None:
        try:
            # Reshape X_test for LSTM input
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            # Predict using LSTM model
            predictions = model.predict(X_test)

            # If model predicts multiple outputs per timestep, take the first column
            if predictions.shape[1] > 1:
                predictions = predictions[:, 0]  

            # Reshape predictions before inverse transformation
            predictions = predictions.reshape(-1, 1)

            # Prepare for inverse transformation
            inverse_input = np.zeros((predictions.shape[0], X_test.shape[1]))
            inverse_input[:, 0] = predictions.flatten()

            # Inverse transform
            predictions_rescaled = scaler.inverse_transform(inverse_input)[:, 0]

            # Generate future dates
            last_date = date_series.max()
            future_dates = pd.date_range(start=last_date, periods=len(predictions_rescaled) + 1, freq="D")[1:]

            # Create DataFrame for display
            predictions_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted Price": predictions_rescaled
            })

            st.subheader("📅 Predicted Future Prices:")
            st.write(predictions_df)

            # Plot predictions
            st.subheader("📊 Price Prediction Graph")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(predictions_df["Date"], predictions_df["Predicted Price"], marker='o', linestyle='-', color='b', label="Predicted Price")
            ax.set_xlabel("Date")
            ax.set_ylabel("Predicted Price")
            ax.set_title("Iron Ore Price Prediction Over Time")
            ax.legend()
            plt.xticks(rotation=45)

            st.pyplot(fig)

        except ValueError as e:
            st.error(f"❌ Model input shape mismatch: {e}")
