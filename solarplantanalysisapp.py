import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split

# Function to load CSV files from GitHub
def load_data(url):
    try:
        data = pd.read_csv(url)
        return data
    except Exception as e:
        st.error(f"Error loading data from {url}: {e}")
        return None

# GitHub raw CSV links
plant_1_generation_url = 'https://raw.githubusercontent.com/Sivatech24/DataSetsForTheModel/0deb87623911b017969be1ab482da725a0ae720c/DataSetsCsvFiles/Plant_1_Generation_Data.csv'
plant_1_weather_url = 'https://raw.githubusercontent.com/Sivatech24/DataSetsForTheModel/0deb87623911b017969be1ab482da725a0ae720c/DataSetsCsvFiles/Plant_1_Weather_Sensor_Data.csv'
plant_2_generation_url = 'https://raw.githubusercontent.com/Sivatech24/DataSetsForTheModel/0deb87623911b017969be1ab482da725a0ae720c/DataSetsCsvFiles/Plant_2_Generation_Data.csv'
plant_2_weather_url = 'https://raw.githubusercontent.com/Sivatech24/DataSetsForTheModel/0deb87623911b017969be1ab482da725a0ae720c/DataSetsCsvFiles/Plant_2_Weather_Sensor_Data.csv'

# Load datasets
st.title('Solar Power Plant Data Overview')

st.subheader('Plant 1 Generation Data')
gen_data = load_data(plant_1_generation_url)
if gen_data is not None:
    st.write(gen_data)

st.subheader('Plant 1 Weather Sensor Data')
weather_data = load_data(plant_1_weather_url)
if weather_data is not None:
    st.write(weather_data)

# Data Processing and Visualization
if gen_data is not None and weather_data is not None:
    # st.subheader('Convert DATE_TIME columns to datetime')
    gen_data['DATE_TIME'] = pd.to_datetime(gen_data['DATE_TIME'], format='%d-%m-%Y %H:%M')
    weather_data['DATE_TIME'] = pd.to_datetime(weather_data['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

    # st.subheader('Resampling generation data daily')
    gen_data_daily = gen_data.set_index('DATE_TIME').resample('D').sum().reset_index()

    st.subheader('Plotting generation data')
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
    gen_data.plot(x='DATE_TIME', y=['DAILY_YIELD', 'TOTAL_YIELD'], ax=ax[0], title="Daily and Total Yield (Generation Data)")
    gen_data.plot(x='DATE_TIME', y=['AC_POWER', 'DC_POWER'], ax=ax[1], title="AC Power & DC Power (Generation Data)")
    st.pyplot(fig)

    st.subheader('Plotting weather data')
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
    weather_data.plot(x='DATE_TIME', y='IRRADIATION', ax=ax[0], title="Irradiation (Weather Data)")
    weather_data.plot(x='DATE_TIME', y=['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE'], ax=ax[1], title="Ambient & Module Temperature (Weather Data)")
    st.pyplot(fig)

    st.subheader('Calculating DC Power Converted')
    gen_data['DC_POWER_CONVERTED'] = gen_data['DC_POWER'] * 0.98  # Assume 2% loss in conversion
    fig, ax = plt.subplots(figsize=(15, 5))
    gen_data.plot(x='DATE_TIME', y='DC_POWER_CONVERTED', ax=ax, title="DC Power Converted")
    st.pyplot(fig)

    st.subheader('Filtering for day time hours')
    day_data_gen = gen_data[(gen_data['DATE_TIME'].dt.hour >= 6) & (gen_data['DATE_TIME'].dt.hour <= 18)]
    fig, ax = plt.subplots(figsize=(15, 5))
    day_data_gen.plot(x='DATE_TIME', y='DC_POWER', ax=ax, title="DC Power Generated During Day Hours")
    st.pyplot(fig)

    st.subheader('Inverter performance analysis')
    inverter_performance = gen_data.groupby('SOURCE_KEY')['DC_POWER'].mean().sort_values()
    st.write(f"Underperforming inverter: {inverter_performance.idxmin()}")

    st.subheader('Inverter specific data')
    inverter_data = gen_data[gen_data['SOURCE_KEY'] == 'bvBOhCH3iADSZry']
    fig, ax = plt.subplots(figsize=(15, 5))
    inverter_data.plot(x='DATE_TIME', y=['AC_POWER', 'DC_POWER'], ax=ax, title="Inverter bvBOhCH3iADSZry")
    st.pyplot(fig)

    st.subheader('Daily yield analysis')
    df_daily_gen = gen_data_daily[['DATE_TIME', 'DAILY_YIELD']].set_index('DATE_TIME')
    result = adfuller(df_daily_gen['DAILY_YIELD'].dropna())
    st.write(f'ADF Statistic: {result[0]}')
    st.write(f'p-value: {result[1]}')

    # st.subheader('Splitting the dataset for ARIMA modeling')
    train_gen, test_gen = train_test_split(df_daily_gen, test_size=0.2, shuffle=False)

    st.subheader('ARIMA model')
    arima_model_gen = ARIMA(train_gen['DAILY_YIELD'], order=(5, 1, 0))
    arima_fit_gen = arima_model_gen.fit()
    forecast_arima_gen = arima_fit_gen.forecast(steps=len(test_gen))
    test_gen['Forecast_ARIMA'] = forecast_arima_gen

    st.subheader('Plotting ARIMA results')
    fig, ax = plt.subplots(figsize=(15, 5))
    train_gen['DAILY_YIELD'].plot(ax=ax, label='Training Data')
    test_gen['DAILY_YIELD'].plot(ax=ax, label='Test Data')
    test_gen['Forecast_ARIMA'].plot(ax=ax, label='ARIMA Forecast')
    plt.legend()
    st.pyplot(fig)

    st.subheader('SARIMA model')
    sarima_model = SARIMAX(train_gen['DAILY_YIELD'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_fit = sarima_model.fit(disp=False)
    sarima_forecast = sarima_fit.forecast(steps=len(test_gen))
    test_gen['Forecast_SARIMA'] = sarima_forecast

    st.subheader('Plotting SARIMA results')
    fig, ax = plt.subplots(figsize=(15, 5))
    train_gen['DAILY_YIELD'].plot(label='Train')
    test_gen['DAILY_YIELD'].plot(label='Test')
    test_gen['Forecast_SARIMA'].plot(label='SARIMA Forecast')
    plt.legend()
    st.pyplot(fig)

    st.subheader('Comparing ARIMA and SARIMA forecasts')
    plt.figure(figsize=(15, 5))
    plt.plot(test_gen.index, test_gen['DAILY_YIELD'], label='Actual Test Data')
    plt.plot(test_gen.index, test_gen['Forecast_ARIMA'], label='ARIMA Forecast')
    plt.plot(test_gen.index, test_gen['Forecast_SARIMA'], label='SARIMA Forecast')
    plt.legend()
    plt.title("ARIMA vs SARIMA Forecast Comparison (Generation Data)")
    st.pyplot(plt)