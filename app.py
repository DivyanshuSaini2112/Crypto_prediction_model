import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import timedelta

from data_loader.indicators import calculate_indicators

# Page config
st.set_page_config(page_title="Real-Time Crypto Predictor", layout="wide")

st.title("📈 Real-Time Bitcoin Price Predictor (Orbit Model)")
st.markdown("""
This dashboard fetches real-time data from Yahoo Finance, computes technical indicators, 
and uses the pre-trained Orbit model to predict tomorrow's Bitcoin price.
""")

# 1. Load Model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error("Failed to load model. Ensure model.pkl exists.")
    st.stop()

# 2. Fetch Live Data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_live_data(symbol="BTC-USD", days=150):
    df = yf.download(symbol, period=f"{days}d", interval="1d")
    if df.empty:
        return df
        
    df = df.reset_index()
    
    # Flatten MultiIndex columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df = df.rename(columns={date_col: 'Date'})
    
    # Rename columns to match what the model expects
    df = df.rename(columns={
        'Open': 'open',
        'Close': 'close',
        'Volume': 'volume',
    })
    
    # We need open, High, Low, close, volume
    df = df[['Date', 'open', 'High', 'Low', 'close', 'volume']]
    
    # Remove timezone information to avoid incompatibility with orbit
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
        
    df = df.dropna().reset_index(drop=True)
    return df

with st.spinner("Fetching live Bitcoin data..."):
    df = get_live_data()

if df.empty:
    st.error("Failed to fetch data from Yahoo Finance.")
    st.stop()

# Show current status
latest_date = df['Date'].iloc[-1].strftime('%Y-%m-%d')
latest_close = df['close'].iloc[-1]
st.metric("Latest Actual Close Price", f"${latest_close:,.2f}", f"As of {latest_date}")

# 3. Calculate Indicators and Prepare Features
def prepare_features(df, look_back=5):
    # Calculate Mean price used for indicators
    df['Mean'] = (df['Low'] + df['High']) / 2
    
    # Calculate all indicators
    indicators = calculate_indicators(
        mean_=np.array(df['Mean']), 
        low_=np.array(df['Low']),
        high_=np.array(df['High']), 
        open_=np.array(df['open']),
        close_=np.array(df['close']), 
        volume_=np.array(df['volume'])
    )
    
    # We only need rsi and macd based on the training config
    df['rsi'] = indicators['rsi']
    df['macd'] = indicators['macd']
    
    # Drop NaNs caused by the indicator lookback periods
    df = df.dropna().reset_index(drop=True)
    
    # The training script adds 'mean' as the final feature
    feature_cols = ['open', 'High', 'Low', 'close', 'volume', 'rsi', 'macd', 'Mean']
    
    if len(df) < look_back:
        raise ValueError("Not enough data to create a prediction window.")
        
    # Get the last `look_back` days (last 5 days)
    # The model was trained with look_back=5, but in train.py, the final day's features 
    # ('predicted_high', 'predicted_low') are completely dropped before training.
    # So the model only expects features from day 0 to day 3.
    tomorrow = df['Date'].iloc[-1] + timedelta(days=1)
    
    last_window = df[feature_cols].tail(look_back).values  # Shape: (5, 8)
    
    row_dict = {'Date': tomorrow}
    
    # Fill days 0 to 3 ONLY (day 4 is entirely dropped in train.py)
    for day in range(look_back - 1):
        for col_idx, col in enumerate(feature_cols):
            # In training, 'Mean' was renamed to 'mean' in the column names
            col_name = 'mean' if col == 'Mean' else col
            row_dict[f"{col_name}_day{day}"] = last_window[day, col_idx]
            
    test_x = pd.DataFrame([row_dict])
    
    return test_x, tomorrow

try:
    test_x, tomorrow = prepare_features(df)
except Exception as e:
    st.error(f"Error calculating features: {e}")
    st.stop()

# 4. Predict
if st.button("Predict Tomorrow's Price", type="primary"):
    with st.spinner("Running Orbit Model Prediction..."):
        try:
            # Predict returns a numpy array
            prediction = model.predict(test_x)
            predicted_price = prediction[0]
            
            st.success(f"### Predicted Price for {tomorrow.strftime('%Y-%m-%d')}: **${predicted_price:,.2f}**")
            
            # Plot the recent history + prediction
            fig = go.Figure()
            
            # Last 30 days of actual close prices
            plot_df = df.tail(30)
            
            fig.add_trace(go.Scatter(
                x=plot_df['Date'], 
                y=plot_df['close'],
                mode='lines+markers',
                name='Actual Close Price',
                line=dict(color='blue')
            ))
            
            # Connect the last actual point to the prediction
            fig.add_trace(go.Scatter(
                x=[plot_df['Date'].iloc[-1], tomorrow],
                y=[plot_df['close'].iloc[-1], predicted_price],
                mode='lines+markers',
                name='Prediction',
                line=dict(color='red', dash='dash'),
                marker=dict(size=10, symbol='star')
            ))
            
            fig.update_layout(
                title="Bitcoin Price Trend & Prediction",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
