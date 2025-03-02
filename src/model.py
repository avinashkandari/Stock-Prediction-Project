from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

def build_lstm_model(input_shape):
    """
    Build the LSTM model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X, y, epochs=10, batch_size=32):
    """
    Train the LSTM model.
    """
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=epochs, batch_size=batch_size)
    return model

def save_model(model, filepath):
    """
    Save the trained model to a file.
    """
    model.save(filepath)
    model.save("models/lstm_model.h5")
