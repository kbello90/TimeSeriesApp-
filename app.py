import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("life.csv", parse_dates=["year"])
    df.set_index("year", inplace=True)
    return df

df = load_data()

# Train ARIMA Model
model = ARIMA(df["value"], order=(1,1,1))
model_fit = model.fit()

# Forecasting Function
def forecast(n):
    return model_fit.forecast(steps=n)

# Streamlit UI
st.title("📈 Time Series Forecasting App")
st.write("This app allows you to visualize past data and forecast future values.")

# Show Data
st.subheader("Historical Data")
st.line_chart(df)

# User Input for Forecasting
n_steps = st.slider("Select number of future years to forecast", min_value=1, max_value=10, value=5)

# Generate Forecast
if st.button("Generate Forecast"):
    forecast_values = forecast(n_steps)
    future_dates = pd.date_range(df.index[-1], periods=n_steps+1, freq="Y")[1:]
    
    # Plot results
    plt.figure(figsize=(10,5))
    plt.plot(df.index, df["value"], label="Historical Data")
    plt.plot(future_dates, forecast_values, label="Forecast", linestyle="dashed", color="red")
    plt.legend()
    st.pyplot(plt)

    # Show Forecasted Values
    forecast_df = pd.DataFrame({"Year": future_dates, "Forecasted Value": forecast_values})
    st.write("### Forecasted Values")
    st.dataframe(forecast_df)

