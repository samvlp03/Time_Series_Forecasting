import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load stock data from CSV file
# Replace 'your_stock_data.csv' with the actual file path or data source
file_path = 'yahoo_stock.csv'
stock_data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Visualize the time series data
stock_data['Close'].plot(figsize=(12, 6))
plt.title('Stock Prices Time Series Data')
plt.show()

# Split the data into training and testing sets
train_size = int(len(stock_data) * 0.8)
train, test = stock_data['Close'][:train_size], stock_data['Close'][train_size:]

# Train the SARIMA model
order = (1, 1, 1)  # Replace with appropriate order (p, d, q) values based on your data
seasonal_order = (1, 1, 1, 12)  # Replace with appropriate seasonal order (P, D, Q, s) values based on your data
model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
fit_model = model.fit()

# Forecast future values
forecast = fit_model.get_forecast(steps=len(test))
forecast_mean = forecast.predicted_mean

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(train, label='Training Data')
plt.plot(test, label='Testing Data')
plt.plot(test.index, forecast_mean, label='SARIMA Forecast', color='red')
plt.title('SARIMA Stock Prices Time Series Forecasting')
plt.legend()
plt.show()

# Evaluate the model performance
mse = mean_squared_error(test, forecast_mean)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test, forecast_mean)

print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
