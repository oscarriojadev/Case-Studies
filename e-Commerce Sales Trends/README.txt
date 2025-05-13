# E-Commerce Sales Trends Analysis

A Python tool for analyzing and forecasting e-commerce sales time series data using ARIMA modeling and decomposition techniques.

## Overview

This tool provides functionality to analyze historical e-commerce sales data and make forecasts for future periods. It implements time series analysis techniques including:

- Time series visualization
- Seasonal decomposition (trend, seasonality, residual)
- Stationarity testing
- Autocorrelation analysis
- ARIMA modeling and forecasting
- Model evaluation

## Requirements

```
pandas
numpy
matplotlib
seaborn
statsmodels
scikit-learn
```

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
   ```

## Usage

1. Prepare your time series data as a CSV file with at least a 'date' column and a numeric value column
2. Modify the `main()` function in `ecommerce_sales_trends.py` with your file path
3. Run the script:
   ```
   python ecommerce_sales_trends.py
   ```

## Function Reference

### `load_data(file_path)`
Loads time series data from a CSV file with proper date parsing.

### `plot_time_series(data, title='Time Series Data')`
Visualizes the time series data with customizable title.

### `decompose_time_series(data, model='additive', period=None)`
Decomposes time series into trend, seasonal, and residual components.

### `test_stationarity(data)`
Tests if the time series is stationary using the Augmented Dickey-Fuller test.

### `plot_acf_pacf(data, lags=None)`
Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF).

### `fit_arima_model(data, order=(1, 1, 1))`
Fits an ARIMA model to the time series data with specified order parameters.

### `forecast_arima(fitted_model, steps=10)`
Forecasts future values using a fitted ARIMA model.

### `evaluate_arima_model(fitted_model, data)`
Evaluates model performance using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

### `plot_forecast(data, forecast, title='Forecast')`
Visualizes actual data alongside forecasted values.

## Example

```python
# Example usage
file_path = 'sales_data.csv'
data = load_data(file_path)

# Visualize the time series
plot_time_series(data)

# Decompose into trend, seasonal, and residual components
decomposition = decompose_time_series(data, model='additive', period=12)

# Check if the time series is stationary
stationarity_result = test_stationarity(data)

# Analyze autocorrelation patterns
plot_acf_pacf(data, lags=20)

# Fit an ARIMA model
fitted_model = fit_arima_model(data, order=(1, 1, 1))

# Make forecasts
forecast_values = forecast_arima(fitted_model, steps=10)
print('Forecasted Values:', forecast_values)

# Evaluate model performance
mse, rmse = evaluate_arima_model(fitted_model, data)

# Visualize forecasts
plot_forecast(data, forecast_values, title='ARIMA Forecast')
```

## Data Format

The script expects a CSV file with:
- A 'date' column that can be parsed into datetime format
- At least one numeric column containing the time series values

Example:
```
date,sales
2023-01-01,1200
2023-01-02,1350
2023-01-03,1100
...
```

## Customization

You can customize the analysis by modifying:
- The ARIMA model order (p, d, q) parameters
- Forecast horizon (steps parameter)
- Seasonal decomposition model (additive or multiplicative)
- Seasonal period for decomposition

## License

[MIT License]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
