# Call Data Analysis and Forecasting

## Overview
This project focuses on analyzing and forecasting call volume data using Python. It generates a synthetic dataset, performs exploratory data analysis (EDA), and applies time series forecasting techniques (SARIMA) to predict future trends.

## Features
- **Synthetic Data Generation**: Generates call data with random attributes.
- **Data Analysis & Visualization**:
  - Time series trends
  - Seasonal decomposition
  - Outlier detection
  - Category-wise distributions
  - Year-over-year (YoY) analysis
- **Forecasting**:
  - SARIMA-based model for call volume prediction
  - RMSE evaluation of model performance
  - Future call volume forecasting
  - Confidence interval estimation
- **Export**: Forecasted results are saved in an Excel file (`forecasted_calls_optimized.xlsx`).

## Prerequisites
Ensure you have the following Python packages installed:
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn openpyxl
```

## Usage
Run the script as follows:
```bash
python script.py
```

## Functions
### `generate_dataset(start_date='2018-01-01', end_date='2024-01-31', num_records=125984)`
Creates a synthetic dataset of calls and saves it as `calls_data.csv`.

### `analyze_data(data, level='day')`
Performs exploratory analysis with visualizations and insights. Available levels: `day`, `week`, `month`.

### `predict_future_numbers(data, level='day', periods=12)`
Forecasts future call volumes using SARIMA and evaluates model performance.

## Data Considerations
- The dataset includes call dates, categories, request types, and durations.
- Incomplete yearly data is flagged during YoY analysis.
- Missing values are handled by forward-filling.

## Output
- **Plots**: Various visualizations are displayed.
- **Excel File**: Forecasted results are saved in `forecasted_calls_optimized.xlsx`.

## Contact
For any queries, please reach out to the developer.

