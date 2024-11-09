import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Function to fetch data
def fetch_data(stock_symbol, start_date, end_date):
    try:
        # Fetch data using yfinance
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function for greeting
def greet_user():
    current_hour = datetime.now().hour
    if current_hour < 12:
        return "Good Morning"
    elif 12 <= current_hour < 18:
        return "Good Afternoon"
    else:
        return "Good Evening"

# Main Streamlit app
def main():
    # Streamlit greeting
    st.title('Stock Price Prediction App')
    greeting = greet_user()
    st.write(f"{greeting}! Welcome to the Stock Price Prediction App 🏦📈")

    # Streamlit inputs for stock symbol and date range
    stock = st.text_input('Enter Stock Symbol', 'AAPL')
    start_date = st.date_input('Start Date', value=pd.to_datetime('2015-01-01'))
    end_date = st.date_input('End Date', value=pd.to_datetime('2025-12-31'))

    # Fetch stock data
    data = fetch_data(stock, start_date, end_date)

    if data is None or data.empty:
        st.error("No data found for the selected stock symbol and date range.")
        return

    st.subheader(f'Stock Data for {stock}')
    st.write(data)

    # Data preprocessing
    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])  # Use the first 80% for training
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])  # The remaining 20% for testing

    # Handle missing values by dropping NaN rows
    data_train = data_train.dropna()
    data_test = data_test.dropna()

    if data_train.empty or data_test.empty:
        st.error("Data after splitting is empty. Please try a different date range.")
        return

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    data_train_scaled = scaler.fit_transform(data_train)
    data_test_scaled = scaler.transform(data_test)

    # Create training and testing datasets for prediction
    X_train = np.array([data_train_scaled[i-60:i, 0] for i in range(60, len(data_train_scaled))])
    y_train = np.array([data_train_scaled[i, 0] for i in range(60, len(data_train_scaled))])

    X_test = np.array([data_test_scaled[i-60:i, 0] for i in range(60, len(data_test_scaled))])
    y_test = np.array([data_test_scaled[i, 0] for i in range(60, len(data_test_scaled))])

    # Reshape data for model
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

    # Model training (Simple Linear Regression for demonstration)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Reverse the scaling for plotting actual vs predicted prices
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))

    # Plotting the results
    st.subheader('Stock Price Prediction')
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_rescaled, color='blue', label='Actual Stock Price')
    plt.plot(y_pred_rescaled, color='red', label='Predicted Stock Price')
    plt.title(f'{stock} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot(plt)

    # Show prediction results
    st.subheader('Prediction Results')
    st.write(f"Predicted stock price: {y_pred_rescaled[-1][0]:.2f}")
    st.write(f"Actual stock price: {y_test_rescaled[-1][0]:.2f}")

    # Predict future prices based on user input for N days
    future_days = st.number_input('Predict Future Days', min_value=1, max_value=365, value=30)

    # Predict the future stock prices
    last_60_days = data.Close[-60:].values.reshape(-1, 1)
    last_60_days_scaled = scaler.transform(last_60_days)

    future_predictions = []

    for i in range(future_days):
        X_input = last_60_days_scaled[-60:].reshape(1, -1)
        future_price = model.predict(X_input)
        future_predictions.append(future_price)
        last_60_days_scaled = np.append(last_60_days_scaled, future_price.reshape(1, 1), axis=0)

    # Reverse the scaling of the predicted future prices
    future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Visualizing the future prediction
    st.subheader(f'Future Stock Price Prediction for {future_days} Days')
    future_dates = pd.date_range(data.index[-1], periods=future_days + 1, freq='D')[1:]

    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data.Close, color='blue', label='Historical Stock Price')
    plt.plot(future_dates, future_predictions_rescaled, color='green', label=f'Predicted Next {future_days} Days')
    plt.title(f'{stock} Future Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot(plt)

    # Model performance (RMSE)
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    st.write(f"Root Mean Squared Error (RMSE) of the model: {rmse:.4f}")

    # Download predicted data (future predictions)
    download_data = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions_rescaled.flatten()})
    st.download_button(
        label="Download Future Predictions",
        data=download_data.to_csv(index=False),
        file_name=f"{stock}_future_predictions.csv",
        mime="text/csv"
    )

# Run the Streamlit app
if __name__ == '__main__':
    main()
