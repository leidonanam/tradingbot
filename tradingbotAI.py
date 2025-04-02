import ccxt
import pandas as pd
import numpy as np
import time
import logging
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Logging setup
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_and_print(msg):
    logging.info(msg)
    print(msg)

# Initialize exchange (dynamic selection)
def initialize_exchange(exchange_name, api_key, api_secret, password=None):
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({
        'apiKey': api_key,
        'secret': api_secret,
        'password': password,
        'enableRateLimit': True,
    })
    return exchange

# User-defined exchange parameters
EXCHANGE_NAME = 'binance'  # Change to preferred exchange
API_KEY = 'YOUR_API_KEY'
API_SECRET = 'YOUR_API_SECRET'
PASSWORD = None  # Some exchanges require a password

exchange = initialize_exchange(EXCHANGE_NAME, API_KEY, API_SECRET, PASSWORD)

# Parameters
symbol = 'BTC/USDT'
timeframe = '5m'
capital = 1000
risk_per_trade = 0.02
stop_loss_percent = 0.01
take_profit_percent = 0.03
short_sma_period = 10
long_sma_period = 50
rsi_period = 14
rsi_overbought = 70
rsi_oversold = 30
lstm_timesteps = 20
max_consecutive_losses = 5  # New: Stop trading after X consecutive losses
trading_fee_percent = 0.1 / 100  # New: 0.1% trading fee

# Fetch OHLCV data
def fetch_data(limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        log_and_print(f"Error fetching data: {e}")
        return None

# Calculate indicators
def calculate_indicators(df):
    df['short_sma'] = df['close'].rolling(window=short_sma_period).mean()
    df['long_sma'] = df['close'].rolling(window=long_sma_period).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df.dropna()

# Prepare data for LSTM
def prepare_lstm_data(df, timesteps=lstm_timesteps):
    df = calculate_indicators(df)
    df['price_change'] = df['close'].pct_change()
    df['target'] = (df['price_change'].shift(-1) > 0).astype(int)
    features = ['close', 'short_sma', 'long_sma', 'rsi', 'price_change']
    df = df.dropna()
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(timesteps, len(scaled_data)):
        X.append(scaled_data[i-timesteps:i])
        y.append(df['target'].iloc[i])
    
    return np.array(X), np.array(y), scaler, features, df.index[timesteps:]

# Train LSTM model
def train_lstm_model(X, y):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    return model

# Position sizing (includes trading fee)
def calculate_position_size(price, capital):
    risk_amount = capital * risk_per_trade
    stop_loss_amount = price * stop_loss_percent
    position_size = risk_amount / (stop_loss_amount * (1 + trading_fee_percent))
    return min(position_size, capital / price)

# Backtest function
def backtest(df, model, scaler, features):
    log_and_print("Starting backtest...")
    df = calculate_indicators(df)
    X, _, _, _, valid_index = prepare_lstm_data(df)
    predictions = model.predict(X, verbose=0)
    df['lstm_signal'] = np.nan
    df.loc[valid_index, 'lstm_signal'] = (predictions > 0.5).astype(int)
    df['lstm_signal'] = df['lstm_signal'].fillna(method='ffill')
    df['prev_lstm_signal'] = df['lstm_signal'].shift(1)
    return df

# Main trading loop
def trading_bot():
    global capital
    position = None
    position_size = 0
    stop_loss = 0
    take_profit = 0
    loss_streak = 0

    df = fetch_data(limit=500)
    X, y, scaler, features, _ = prepare_lstm_data(df)
    model = train_lstm_model(X, y)
    df = backtest(df, model, scaler, features)
    log_and_print("Backtest complete.")
    
    while True:
        try:
            df = fetch_data(limit=100)
            if df is None:
                time.sleep(60)
                continue

            df = calculate_indicators(df)
            current_price = df['close'].iloc[-1]
            rsi = df['rsi'].iloc[-1]
            lstm_signal = model.predict(np.array([[current_price]]))[0][0] > 0.5
            
            if position == 'long' and (current_price <= stop_loss or current_price >= take_profit):
                capital += position_size * current_price * (1 - trading_fee_percent)
                log_and_print(f"Trade closed at {current_price:.2f}")
                position, position_size = None, 0
                loss_streak = 0 if current_price >= take_profit else loss_streak + 1
            else:
                position_size = calculate_position_size(current_price, capital)
                position, position_size, stop_loss, take_profit = execute_trade(lstm_signal, rsi, current_price, position_size, loss_streak)
                if position:
                    capital -= position_size * current_price * (1 + trading_fee_percent)
            
            log_and_print(f"Capital: {capital:.2f} USDT")
            time.sleep(60)
        except Exception as e:
            log_and_print(f"Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    log_and_print("Starting Trading Bot...")
    trading_bot()
