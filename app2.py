import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import mysql.connector
from sklearn.preprocessing import MinMaxScaler

# Function to establish MySQL Database Connection
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="user1",
            password="user1",
            database="iron_ore"
        )
        return conn
    except mysql.connector.Error as e:
        st.error(f"❌ Database Connection Failed: {e}")
        return None

# User Authentication
users = {"admin": "password123", "user": "userpass"}  # Example users

st.sidebar.title("🔑 User Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_button = st.sidebar.button("Login")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if login_button:
    if username in users and users[username] == password:
        st.sidebar.success("✅ Login Successful!")
        st.session_state.logged_in = True
    else:
        st.sidebar.error("❌ Invalid Username or Password")
        st.session_state.logged_in = False

# Load LSTM Model
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

try:
    model = tf.keras.models.load_model("lstm_model.h5", custom_objects={"mse": mse})
    st.success("✅ LSTM Model Loaded Successfully!")
    
    # Display Model Summary
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    st.subheader("📜 Model Summary")
    st.code("\n".join(model_summary), language="text")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Function to load and preprocess test data
def load_and_preprocess_test_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    date_column = [col for col in df.columns if "date" in col.lower()]
    
    if not date_column:
        st.error("❌ No date column found. Ensure the CSV has a valid date column.")
        return None, None, None

    date_column = date_column[0]
    df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, errors="coerce")
    df.dropna(inplace=True)

    numerical_cols = df.select_dtypes(include=["number"]).columns
    
    if numerical_cols.empty:
        st.error("❌ No numerical columns found for prediction.")
        return None, None, None

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[numerical_cols])

    return df_scaled, scaler, df[date_column]

# Streamlit UI
st.title("🔮 Iron Ore Price Prediction")

if st.session_state.logged_in:
    uploaded_file = st.file_uploader("📂 Upload your test dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        X_test, scaler, date_series = load_and_preprocess_test_data(uploaded_file)

        if X_test is not None:
            try:
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                predictions = model.predict(X_test)

                if predictions.shape[1] > 1:
                    predictions = predictions[:, 0]  

                predictions = predictions.reshape(-1, 1)
                inverse_input = np.zeros((predictions.shape[0], X_test.shape[1]))
                inverse_input[:, 0] = predictions.flatten()
                predictions_rescaled = scaler.inverse_transform(inverse_input)[:, 0]

                last_date = date_series.max()
                future_dates = pd.date_range(start=last_date, periods=len(predictions_rescaled) + 1, freq="D")[1:]
                
                # Confidence Interval Input
                confidence_interval = st.slider("Select Confidence Interval (%)", 50, 99, 95)
                margin_of_error = (100 - confidence_interval) / 100 * predictions_rescaled
                lower_bound = predictions_rescaled - margin_of_error
                upper_bound = predictions_rescaled + margin_of_error
                
                predictions_df = pd.DataFrame({
                    "Date": future_dates.strftime('%Y-%m-%d'),
                    "Predicted Price": predictions_rescaled.astype(float),
                    "Lower Bound": lower_bound.astype(float),
                    "Upper Bound": upper_bound.astype(float),
                    "Confidence Interval": confidence_interval
                })

                st.subheader("📅 Predicted Future Prices:")
                st.write(predictions_df)
                
                # Plot predictions
                st.subheader("📊 Price Prediction Graph")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(predictions_df["Date"], predictions_df["Predicted Price"], marker='o', linestyle='-', color='b', label="Predicted Price")
                ax.fill_between(predictions_df["Date"], predictions_df["Lower Bound"], predictions_df["Upper Bound"], color='b', alpha=0.2, label="Confidence Interval")
                ax.set_xlabel("Date")
                ax.set_ylabel("Predicted Price")
                ax.set_title("Iron Ore Price Prediction Over Time")
                ax.legend()
                plt.xticks(rotation=45)
                
                st.pyplot(fig)
                
                # Save Predictions to Database
                if st.button("💾 Save Predictions"):
                    conn = get_db_connection()
                    if conn:
                        cursor = conn.cursor()
                        for i in range(len(predictions_df)):
                            cursor.execute("""
                                INSERT INTO predictions (username, date, predicted_price, lower_bound, upper_bound, confidence_interval)
                                VALUES (%s, %s, %s, %s, %s, %s)
                            """, (
                                username, 
                                predictions_df.iloc[i]["Date"], 
                                float(predictions_df.iloc[i]["Predicted Price"]), 
                                float(predictions_df.iloc[i]["Lower Bound"]), 
                                float(predictions_df.iloc[i]["Upper Bound"]), 
                                int(confidence_interval)
                            ))
                        conn.commit()
                        cursor.close()
                        conn.close()
                        st.success("✅ Predictions Saved Successfully!")
                    else:
                        st.error("❌ Failed to save predictions due to database connection issues.")
                
            except ValueError as e:
                st.error(f"❌ Model input shape mismatch: {e}")
else:
    st.warning("🔐 Please log in to use the prediction system.")
