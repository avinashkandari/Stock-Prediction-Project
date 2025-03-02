
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.data_fetcher import fetch_stock_data, fetch_news_articles, analyze_sentiment
from src.preprocessor import preprocess_data
from src.model import build_lstm_model
from src.predictor import predict_next_10_days

def evaluate_model(actual, predicted):
    """
    Evaluate the model's performance using RMSE and MAE.
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

def main():
    # Parameters
    ticker = input("Enter stock ticker symbol (e.g., AAPL): ")
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    eodhd_api_key = '67c3793e67ebc4.02813409'  # Replace with your EODHD API key

    # Fetch data
    stock_prices, historical_dates = fetch_stock_data(ticker, start_date, end_date, eodhd_api_key)
    if stock_prices is None:
        print("Failed to fetch stock data. Exiting.")
        return

    articles = fetch_news_articles(eodhd_api_key, ticker)
    sentiment_scores = analyze_sentiment(articles)

    # Ensure sentiment_scores has the same length as stock_prices
    if len(sentiment_scores) < len(stock_prices):
        sentiment_scores = np.pad(sentiment_scores, ((0, len(stock_prices) - len(sentiment_scores)), (0, 0)), mode='constant')

    # Check if data is valid
    if stock_prices is None or sentiment_scores is None:
        print("Invalid data. Exiting.")
        return

    # Preprocess data
    X, y, scaler = preprocess_data(stock_prices, sentiment_scores)

    # Build and train LSTM model
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, batch_size=32, epochs=10)

    # Predict prices for each day
    predicted_prices = model.predict(X)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    predicted_prices = predicted_prices.flatten()  # Flatten the array

    # Evaluate model performance
    print("\nModel Evaluation:")
    evaluate_model(stock_prices[60:], predicted_prices)

    # Predict next 10 days
    last_sequence = np.hstack((X[-1, :, 0].reshape(-1, 1), sentiment_scores[-60:].reshape(-1, 1)))  # Last 60 days of data
    next_10_days_predictions = predict_next_10_days(model, last_sequence, scaler)

    # Print predictions
    print("\nPredicted vs Actual Stock Prices:")
    for i in range(len(predicted_prices)):
        print(f"Date: {historical_dates[i + 60]}, Actual: ${stock_prices[i + 60][0]:.2f}, Predicted: ${predicted_prices[i]:.2f}")

    print("\nPredicted Stock Prices for the Next 10 Days:")
    future_dates = [historical_dates[-1] + timedelta(days=i) for i in range(1, 11)]
    for i, price in enumerate(next_10_days_predictions, 1):
        print(f"Date: {future_dates[i - 1].strftime('%Y-%m-%d')}, Predicted: ${price:.2f}")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(historical_dates[60:], stock_prices[60:], label='Actual Prices')
    plt.plot(historical_dates[60:], predicted_prices, label='Predicted Prices')
    plt.plot(future_dates, next_10_days_predictions, 'ro-', label='Next 10 Days Predictions')
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()