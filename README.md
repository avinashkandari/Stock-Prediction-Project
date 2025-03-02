stock-price-prediction/
│
├── data/                   # Folder for storing datasets
│   └── stock_data.csv      # Example: Historical stock data
│
├── models/                 # Folder for saving trained models
│   └── lstm_model.h5       # Example: Saved LSTM model
│
├── src/                    # Source code folder
│   ├── __init__.py         # Makes src a Python package
│   ├── data_fetcher.py     # Fetch stock data and news articles
│   ├── preprocessor.py     # Preprocess data for the model
│   ├── model.py            # Build, train, and save the LSTM model
│   ├── predictor.py        # Predict future stock prices
│   └── main.py             # Main script to run the project
│
├── tests/                  # Unit tests (optional)
│   └── test_model.py       # Example: Test for the LSTM model
│
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation


How to Run the Project
Install Dependencies:Run these 2 command in terminal


1.pip install -r requirements.txt


2.python src/main.py
