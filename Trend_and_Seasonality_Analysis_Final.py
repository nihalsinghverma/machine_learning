import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import DateOffset

# Suppress warnings
warnings.filterwarnings("ignore")

def generate_dataset(start_date='2018-01-01', end_date='2024-01-31', num_records=125984):
    np.random.seed(42)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    call_dates = np.random.choice(date_range, num_records)
    call_categories = np.random.choice(['PO', 'D', 'FO'], num_records)
    request_types = np.random.choice(['Load Change', 'Payments Issue', 'FS Settlement', 'Detention'], num_records)
    call_ids = np.arange(1, num_records + 1)
    
    # Generate random call durations (in minutes)
    call_durations = np.random.randint(1, 60, num_records)  # Random durations between 1 and 60 minutes
    
    df = pd.DataFrame({
        'call_id': call_ids,
        'call_date': call_dates,
        'call_category': call_categories,
        'request_type': request_types,
        'call_duration': call_durations  # Add call duration to the dataset
    })
    df.sort_values('call_date', inplace=True)
    df.to_csv('calls_data.csv', index=False)
    print("Synthetic dataset 'calls_data.csv' generated successfully.")
    return df


def analyze_data(data, level='day'):
    """
    Analyze call data to show trends, generate insights, and visualize distributions.

    Args:
        data (pd.DataFrame): DataFrame with columns 'call_date', 'call_category', and 'request_type'.
        level (str): Aggregation level ('day', 'week', 'month').

    Returns:
        None: Displays plots and prints insights.
    """

    # Convert call_date to datetime
    data['call_date'] = pd.to_datetime(data['call_date'])

    # Handle missing values
    if data.isnull().sum().sum() > 0:
        print("⚠️ Warning: Missing values found! Filling with mode for categorical and median for numerical.")
        data.fillna(method='ffill', inplace=True)

    # Aggregate data based on the level
    freq_map = {'day': 'D', 'week': 'W', 'month': 'M'}
    if level not in freq_map:
        raise ValueError("Invalid level. Choose from 'day', 'week', or 'month'.")

    data_agg = data.groupby(pd.Grouper(key='call_date', freq=freq_map[level])).size()

    # --- Plot 1: Time Series of Call Volume ---
    plt.figure(figsize=(12, 6))
    plt.plot(data_agg, marker='o', linestyle='-', color='blue', label='Call Volume')
    plt.title(f"Call Volume Over Time ({level.capitalize()})")
    plt.xlabel('Date')
    plt.ylabel('Number of Calls')
    plt.legend()
    plt.grid()
    plt.show()

    # --- Decomposing Time Series ---
    decomposition = seasonal_decompose(data_agg, model='additive', period=12)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    decomposition.observed.plot(ax=axes[0], title="Observed")
    decomposition.trend.plot(ax=axes[1], title="Trend")
    decomposition.seasonal.plot(ax=axes[2], title="Seasonality")
    decomposition.resid.plot(ax=axes[3], title="Residuals")
    plt.tight_layout()
    plt.show()

    # --- Insights ---
    print("📊 Insights:")
    print(f"✅ Total calls: {data_agg.sum()}")
    print(f"✅ Average calls per {level}: {data_agg.mean():.2f}")
    print(f"✅ Peak call volume occurred on: {data_agg.idxmax()} with {data_agg.max()} calls.")
    print(f"✅ Lowest call volume occurred on: {data_agg.idxmin()} with {data_agg.min()} calls.")

    # --- Plot 3: Call Volume Distribution ---
    plt.figure(figsize=(10, 5))
    sns.histplot(data_agg, bins=30, kde=True, color='purple')
    plt.title("Distribution of Call Volume")
    plt.xlabel("Number of Calls")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

    # Outlier Detection
    Q1, Q3 = np.percentile(data_agg.dropna(), [25, 75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = data_agg[(data_agg < lower_bound) | (data_agg > upper_bound)]
    
    if not outliers.empty:
        print("Warning ⚠️: Outliers detected in call volume:")
        print(outliers)
        
    # --- Plot 4: Outlier Detection using Boxplot ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=data_agg, color='red')
    plt.title("Outlier Detection in Call Volume")
    plt.xlabel("Number of Calls")
    plt.grid()
    plt.show()

    # --- Plot 5: Call Category Pie Chart ---
    plt.figure(figsize=(8, 6))
    category_counts = data['call_category'].value_counts()
    plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title("Call Category Distribution")
    plt.show()

    # --- Plot 6: Request Type Distribution ---
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x='request_type', order=data['request_type'].value_counts().index, palette='Greens_r')
    plt.title('Request Type Distribution')
    plt.xlabel('Request Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

    # --- Plot 7: Year-over-Year (YoY) Call Volume ---

    # Year-over-Year (YoY) Analysis
    data['Year'] = data['call_date'].dt.year
    yearly_counts = data.groupby('Year').size()

    # Check for incomplete years
    min_year, max_year = data['Year'].min(), data['Year'].max()
    complete_years = []
    
    for year in range(min_year, max_year + 1):
        if data[data['Year'] == year]['call_date'].nunique() >= 365:  # Checking if a full year's data exists
            complete_years.append(year)
        else:
            print(f"Warning ⚠️: Data for the year {year} is incomplete.")

    # Plot YoY bar chart
    if complete_years:
        plt.figure(figsize=(8, 6))
        yearly_counts.loc[complete_years].plot(kind='bar', color='green', alpha=0.7)
        plt.title('Year-over-Year Call Volume')
        plt.xlabel('Year')
        plt.ylabel('Number of Calls')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.show()
    else:
        print("No complete yearly data available for YoY analysis.")

    print("Info: Bar plot for the each year:")
    
    data['year'] = data['call_date'].dt.year
    yoy_counts = data.groupby('year').size()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=yoy_counts.index, y=yoy_counts.values, palette='Blues')
    plt.title("Year-over-Year (YoY) Call Volume")
    plt.xlabel("Year")
    plt.ylabel("Total Calls")
    plt.grid()
    plt.show()

def predict_future_numbers(data, level='day', periods=12):
    """
    Forecast future call volumes using SARIMA, evaluate model performance, and visualize results.

    Args:
    data (pd.DataFrame):     Dataset with a 'call_date' column for time series data.
    level (str, optional):   Aggregation level ('day', 'week', or 'month'). Default is 'day'.
    periods (int, optional): Number of future periods to forecast. Default is 12.

    Returns:
    - RMSE for model performance on test data.
    - Plots: Historical data with forecast, residual distribution, ACF, and PACF.
    - Excel file ('forecasted_calls_optimized.xlsx') with forecasted values and confidence intervals.
    """

    try:
        # Validate level input
        freq_map = {'day': 'D', 'week': 'W', 'month': 'M'}
        offset_map = {'day': 'days', 'week': 'weeks', 'month': 'months'}
        if level not in freq_map:
            raise ValueError("Invalid level. Choose from 'day', 'week', or 'month'.")

        # Convert call_date to datetime and aggregate data
        data['call_date'] = pd.to_datetime(data['call_date'])
        data_agg = data.groupby(pd.Grouper(key='call_date', freq=freq_map[level])).size()

        # Train-Test Split (80% Train, 20% Test)
        train_size = int(len(data_agg) * 0.8)
        train, test = data_agg[:train_size], data_agg[train_size:]

        # Fit SARIMA model
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit(disp=False)

        # Predict on test data
        predictions = results.predict(start=len(train), end=len(data_agg) - 1)
        model_rmse = rmse(test, predictions)
        print(f"Model RMSE: {model_rmse:.2f}")

        # Generate future dates
        last_date = data_agg.index[-1]
        future_dates = [last_date + DateOffset(**{offset_map[level]: i}) for i in range(1, periods + 1)]

        # Forecast future values
        forecast = results.get_forecast(steps=periods)
        forecast_values = forecast.predicted_mean
        forecast_ci = forecast.conf_int()

        # Plot Historical Data & Forecast
        plt.figure(figsize=(12, 6))
        plt.plot(data_agg, label='Historical Data', color='blue')
        plt.plot(test.index, predictions, label='Predictions (Test Data)', color='red')
        plt.plot(future_dates, forecast_values, label='Forecast', color='orange')
        plt.fill_between(future_dates, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='orange', alpha=0.2)
        plt.title(f"Forecasted Call Volume ({level.capitalize()})")
        plt.xlabel('Date')
        plt.ylabel('Number of Calls')
        plt.legend()
        plt.grid()
        plt.show()

        # Plot Residuals
        residuals = train - results.fittedvalues
        plt.figure(figsize=(12, 6))
        sns.histplot(residuals.dropna(), kde=True, bins=30, color='purple')
        plt.title("Residual Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.grid()
        plt.show()

        # Plot ACF & PACF
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_acf(residuals.dropna(), ax=axes[0])
        plot_pacf(residuals.dropna(), ax=axes[1])
        axes[0].set_title("Autocorrelation (ACF)")
        axes[1].set_title("Partial Autocorrelation (PACF)")
        plt.show()

        # Save forecast to Excel
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecasted Calls': forecast_values,
            'Lower CI': forecast_ci.iloc[:, 0],
            'Upper CI': forecast_ci.iloc[:, 1]
        })
        forecast_df.to_excel('forecasted_calls_optimized.xlsx', index=False)
        print("Forecast saved to 'forecasted_calls_optimized.xlsx'.")

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Generate synthetic dataset
    df = generate_dataset()

    # Analyze data
    analyze_data(df, level='day')

    # Predict future call volumes
    predict_future_numbers(df, level='day', periods=12)