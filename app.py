import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st

start = '2009-12-31'
end = '2025-09-30'

st.title('Stock Trend Prediction')

# user_input = st.text_input('Enter Stock Ticker', 'AAPL')
st.subheader("Choose Stock Input Method")
option = st.radio("Select Input Type:", ["Dropdown", "Manual Entry"])

if option == "Dropdown":
    stock_list = {
        "SBIN.NS" : "State Bank Of India",
        "POWERGRID.NS" : "Power Grid Corporation Of India Ltd",
        "NTPC.NS" : "NTPC Ltd",
        "HINDUNILVR.NS" : "Hindustan Unilever Ltd",
        "TATACONSUM.NS" : "Tata Consumer Products Ltd",
        "NESTLEIND.NS" : "Nestle India Ltd",
        "JSWSTEEL.NS" : "JSW Steel Ltd",
        "MARUTI.NS" : "Maruti Suzuki India Ltd",
        "HCLTECH.NS" : "HCL Technologies Ltd",
        "ONGC.NS" : "Oil & Natural Gas Corporation Ltd",
        "CIPLA.NS" : "Cipla Ltd",
        "LT.NS" : "Larsen & Toubro Ltd",
        "ASIANPAINT.NS" : "Asian Paints Ltd",
        "BAJFINANCE.NS" : "Bajaj Finance Ltd",
        "EICHERMOT.NS" : "Eicher Motors Ltd",
        "TRENT.NS" : "Trent Ltd",
        "INFY.NS" : "Infosys Ltd",
        "GRASIM.NS" : "Grasim Industries Ltd",
        "SBILIFE.NS" : "SBI Life Insurance Company Ltd",
        "SUNPHARMA.NS" : "Sun Pharmaceutical Industries Ltd",
        "TATASTEEL.NS" : "Tata Steel Ltd",
        "TATAMOTORS.NS" : "Tata Motors Ltd",
        "ITC.NS" : "ITC Ltd",
        "RELIANCE.NS" : "Reliance Industries Ltd",
        "HDFCLIFE.NS" : "HDFC Life Insurance Company Ltd",
        "HINDALCO.NS" : "Hindalco Industries Ltd",
        "HDFCBANK.NS" : "HDFC Bank Ltd",
        "HEROMOTOCO.NS" : "Hero MotoCorp Ltd",
        "ULTRACEMCO.NS" : "Ultratech Cement Ltd",
        "COALINDIA.NS" : "Coal India Ltd",
        "KOTAKBANK.NS" : "Kotak Mahindra Bank Ltd",
        "BHARTIARTL.NS" : "Bharti Airtel Ltd",
        "AXISBANK.NS" : "Axis Bank Ltd",
        "ICICIBANK.NS" : "ICICI Bank Ltd",
        "ADANIPORTS.NS" : "Adani Ports and Special Economic Zone Ltd",
        "SHRIRAMFIN.NS" : "Shriram Finance Ltd",
        "ADANIENT.NS" : "Adani Enterprises Ltd",
        "BAJAJ-AUTO.NS" : "Bajaj Auto Ltd",
        "TITAN.NS" : "Titan Company Ltd",
        "BAJAJFINSV.NS" : "Bajaj Finserv Ltd",
        "APOLLOHOSP.NS" : "Apollo Hospitals Enterprise Ltd ",
        "DRREDDY.NS" : "Dr. Reddys Laboratories Ltd",
        "BEL.NS" : "Bharat Electronics Ltd",
        "JIOFIN.NS" : "JIO Financial Services Ltd",
        "TCS.NS" : "Tata Consultancy Services Ltd",
        "TECHM.NS" : "Tech Mahindra Ltd",
        "WIPRO.NS" : "Wipro Ltd",
        "ETERNAL.NS" : "Eternal Ltd",
        "M&M.NS" : "Mahindra & Mahindra Ltd",
        "INDUSINDBK.NS" : "IndusInd Bank Ltd",
                }

    # Display format: "Ticker - Full Name"
    stock_display = [f"{ticker} - {name}" for ticker, name in stock_list.items()]
    selected_stock = st.selectbox("Select Stock", stock_display)

    # Extract only ticker (before " - ")
    user_input = selected_stock.split(" - ")[0]

else:
    user_input = st.text_input("Enter Stock Ticker", "SBIN.NS")


# Download data with Adj Close
df = yf.download(user_input, start, end)

# # Reset index to make 'Date' a column
# df = df.reset_index()

# Reorder columns: Date, Open, High, Low, Close, Volume, Adj Close
df = df[['High', 'Low', 'Open', 'Close', 'Volume']]

# Describing Data
st.subheader('Data from 2010 - 2025')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

# Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

# Load my model
model = load_model('keras_model.h5')

# Testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

