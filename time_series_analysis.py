# ============================================================
# LEVEL 3 — TASK 1: TIME SERIES ANALYSIS AND FORECASTING
# ============================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

# Suppress harmless convergence warnings from statsmodels optimiser
warnings.filterwarnings("ignore")


# ============================================================
# STAGE 1 — SETUP
# ============================================================

def create_folders():
    # --------------------------------------------------------
    # Creates output folders if they do not already exist.
    # exist_ok=True means no error is raised if the folder is
    # already there from a previous run.
    # --------------------------------------------------------
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)


# ============================================================
# STAGE 2 — DATA LOADING (yfinance)
# ============================================================

def load_data(ticker, start_date, end_date):
    # --------------------------------------------------------
    # Downloads daily closing prices from Yahoo Finance.
    # Returns a DataFrame with a clean datetime index and a
    # column named 'price'.
    # --------------------------------------------------------
    print(f"Downloading {ticker} daily data from {start_date} to {end_date}...")
    df_raw = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df_raw.empty:
        print("ERROR: No data downloaded. Check ticker or internet connection.")
        return None

    # Keeps only the 'Close' price most common for forecasting
    df = df_raw[['Close']].copy()
    df.columns = ['price']
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Sets frequency to daily calendar days. Weekends will be NaN initially.
    df = df.asfreq('D')
    # Forward fill weekends (Sat/Sun get Friday's closing price)
    df['price'] = df['price'].ffill()

    print(f"SUCCESS: Downloaded {len(df)} observations (weekends filled).")
    print(f"         Date range: {df.index[0].date()} to {df.index[-1].date()}")
    return df


def summarise_data(df):
    # --------------------------------------------------------
    # Prints basic statistics and information about the time series.
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STAGE 2 — DATA SUMMARY")
    print("=" * 60)

    print(f"\n  Total observations : {len(df)}")
    print(f"  Start date         : {df.index[0].date()}")
    print(f"  End date           : {df.index[-1].date()}")
    print(f"  Frequency          : Daily (calendar days, weekends filled)")

    print("\n  --- Price Statistics ---")
    stats = df['price'].describe()
    for label, value in stats.items():
        print(f"  {label:<10}: {value:.2f}")

    print("\n" + "=" * 60)


# ============================================================
# STAGE 3 — RAW TIME SERIES PLOT
# ============================================================

def plot_raw_series(df):
    # --------------------------------------------------------
    # Plot the raw closing prices over time.
    # This is always the first step in any time series project
    # we need to see the data before we can model it.
    # Looking at this plot tells us:
    #   - Is there an upward or downward trend?
    #   - Are there repeating weekly patterns?
    #   - Are there any obvious anomalies or missing data?
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df.index, df['price'], color="#2563EB", linewidth=1.2, label="Daily Closing Price")

    ax.set_title("Apple (AAPL) Daily Closing Price (2020-2024)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = "outputs/plots/01_raw_series.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {output_path}")


# ============================================================
# STAGE 4 — STATIONARITY CHECK
# ============================================================

def check_stationarity(series):
    # --------------------------------------------------------
    # A time series is stationary when its mean, variance, and
    # autocorrelation do not change over time.
    # Most forecasting models (including ARIMA/SARIMA) require
    # the data to be stationary before fitting.
    #
    # We use the Augmented Dickey-Fuller (ADF) test to check.
    # The null hypothesis is that the series is NOT stationary.
    # If the p-value is below 0.05, we reject that hypothesis
    # and conclude the series IS stationary.
    # --------------------------------------------------------
    print("\n  --- Augmented Dickey-Fuller Stationarity Test ---\n")

    result = adfuller(series)
    adf_stat = result[0]
    p_value  = result[1]

    print(f"  ADF Statistic : {adf_stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")

    if p_value < 0.05:
        print("  Result        : Series is STATIONARY (p < 0.05)")
    else:
        print("  Result        : Series is NOT stationary (p >= 0.05)")
        print("                  Differencing will be applied inside the SARIMA model.")


# ============================================================
# STAGE 5 — DECOMPOSITION (TREND, SEASONALITY and RESIDUAL)
# ============================================================

def decompose_series(df):
    # --------------------------------------------------------
    # Decomposition splits a time series into three parts:
    #
    #   Trend      — the long-term direction (going up or down)
    #   Seasonality — repeating patterns at fixed intervals
    #                 (e.g. weekly cycles)
    #   Residual   — whatever is left after removing trend and
    #                seasonality (random noise)
    #
    # We use an ADDITIVE model because stock price movements
    # are roughly additive the seasonal amplitude does not
    # grow strongly with the trend.
    # period=5 tells the function that one full seasonal cycle
    # takes 5 trading days one business week.
    # --------------------------------------------------------
    decomposition = seasonal_decompose(df['price'], model='additive', period=5)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # Original series
    axes[0].plot(decomposition.observed, color="#2563EB", linewidth=1.2)
    axes[0].set_ylabel("Observed", fontsize=10)
    axes[0].set_title("Time Series Decomposition (Additive, Period=5 days)", fontsize=13, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Trend component
    axes[1].plot(decomposition.trend, color="#16A34A", linewidth=1.5)
    axes[1].set_ylabel("Trend", fontsize=10)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    # Seasonal component
    axes[2].plot(decomposition.seasonal, color="#EA580C", linewidth=1.2)
    axes[2].set_ylabel("Seasonal (Weekly)", fontsize=10)
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)

    # Residual component
    axes[3].plot(decomposition.resid, color="#7C3AED", linewidth=1.0)
    axes[3].set_ylabel("Residual", fontsize=10)
    axes[3].set_xlabel("Year", fontsize=10)
    axes[3].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_path = "outputs/plots/02_decomposition.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {output_path}")

    return decomposition


# ============================================================
# STAGE 6 — MOVING AVERAGE SMOOTHING
# ============================================================

def apply_moving_average(df):
    # --------------------------------------------------------
    # A moving average smooths out short-term noise by
    # replacing each data point with the average of its
    # surrounding window of values.
    #
    # We calculate three common windows:
    #   5-day   — one trading week
    #   20-day  — one trading month
    #   50-day  — approximately one quarter
    # --------------------------------------------------------
    df['MA_5']  = df['price'].rolling(window=5).mean()
    df['MA_20'] = df['price'].rolling(window=20).mean()
    df['MA_50'] = df['price'].rolling(window=50).mean()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df.index, df['price'], color="#CBD5E1", linewidth=1.0, alpha=0.7, label="Original")
    ax.plot(df.index, df['MA_5'],  color="#F59E0B", linewidth=1.5, label="5-day MA")
    ax.plot(df.index, df['MA_20'], color="#DC2626", linewidth=1.5, label="20-day MA")
    ax.plot(df.index, df['MA_50'], color="#7C3AED", linewidth=2.0, label="50-day MA")

    ax.set_title("Moving Averages – Apple Stock Price", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = "outputs/plots/03_moving_averages.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {output_path}")

# ============================================================
# STAGE 7 — EXPONENTIAL SMOOTHING (HOLT-WINTERS)
# ============================================================

def apply_exponential_smoothing(df):
    # --------------------------------------------------------
    # Exponential smoothing gives more weight to recent values.
    # Holt-Winters triple exponential smoothing models level,
    # trend and seasonality simultaneously.
    #
    # We use:
    #   trend='add'     — additive trend
    #   seasonal='add'  — additive seasonality (amplitude constant)
    #   seasonal_periods=5 — weekly cycle (5 trading days)
    # --------------------------------------------------------
    model = ExponentialSmoothing(
        df['price'],
        trend='add',
        seasonal='add',
        seasonal_periods=5
    )
    fitted_model = model.fit(optimized=True)
    smoothed = fitted_model.fittedvalues

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df.index, df['price'], color="#CBD5E1", linewidth=1.2, alpha=0.7, label="Original")
    ax.plot(df.index, smoothed,         color="#7C3AED", linewidth=2.0, label="Holt-Winters Smoothing")

    ax.set_title("Holt‑Winters Exponential Smoothing", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = "outputs/plots/04_exponential_smoothing.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {output_path}")

    return fitted_model


# ============================================================
# STAGE 8 — TRAIN / TEST SPLIT
# ============================================================

def split_data(df, test_days=60):
    # --------------------------------------------------------
    # Before building the SARIMA model we split our data into:
    #   Train set — the historical data the model learns from
    #   Test set  — the final N days we hold back so we can
    #               measure how well the model predicts values
    #               it has never seen before.
    #
    # We use the last 60 days ≈2 trading months as the test set.
    # Everything before that is used for training.
    # --------------------------------------------------------
    train = df.iloc[:-test_days]
    test  = df.iloc[-test_days:]

    print(f"\n  Train set: {len(train)} days ({train.index[0].date()} to {train.index[-1].date()})")
    print(f"  Test set : {len(test)} days  ({test.index[0].date()} to {test.index[-1].date()})")
    return train, test


# ============================================================
# STAGE 9 — SARIMA MODEL 
# ============================================================

def build_sarima_model(train):
    # --------------------------------------------------------
    # SARIMA = Seasonal AutoRegressive Integrated Moving Average
    #
    # It is an extension of ARIMA that also handles seasonality.
    # The model has two sets of parameters:
    #
    # Non-seasonal order (p, d, q):
    #   p — number of autoregressive lags
    #   d — degree of differencing 1 makes the series stationary
    #   q — number of moving average terms
    #
    # Seasonal order (P, D, Q, m):
    #   P — seasonal autoregressive lags
    #   D — seasonal differencing
    #   Q — seasonal moving average terms
    #   m — length of one seasonal cycle (5 for trading week)
    #
    # We use (1,1,1)(1,1,1,5) as a starting point for daily
    # financial data with weekly seasonality.
    # --------------------------------------------------------
    print("\n  Fitting SARIMA(1,1,1)(1,1,1,5) model...")

    sarima_model = SARIMAX(
        train['price'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 5),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarima_result = sarima_model.fit(disp=False)

    print("  SARIMA model fitted successfully.")
    print(f"\n  AIC: {sarima_result.aic:.2f}  |  BIC: {sarima_result.bic:.2f}")
    print("  (Lower AIC/BIC = better model fit relative to complexity)")
    return sarima_result


# ============================================================
# STAGE 10 — FORECAST AND EVALUATION 
# ============================================================

def forecast_and_evaluate(sarima_result, train, test, df):
    # --------------------------------------------------------
    # Now we use the trained SARIMA model to forecast values
    # for the test period and evaluate how accurate it was.
    #
    # .get_forecast(steps=N) generates N predictions beyond
    # the end of the training data which lines up with our
    # test set.
    #
    # .conf_int() returns the 95% confidence interval.
    # --------------------------------------------------------
    forecast_object = sarima_result.get_forecast(steps=len(test))
    forecast_mean = forecast_object.predicted_mean
    conf_int = forecast_object.conf_int()

    # --- RMSE ---
    # Root Mean Squared Error measures average forecast error.
    # We square the errors to penalise large misses more,
    # take the mean, then take the square root to get back
    # to the original units (USD).
    rmse = np.sqrt(mean_squared_error(test['price'], forecast_mean))

    # --- MAPE ---
    # Mean Absolute Percentage Error expresses accuracy as a
    # percentage, making it easy to interpret regardless of scale.
    mape = np.mean(np.abs((test['price'] - forecast_mean) / test['price'])) * 100

    print(f"\n  --- Forecast Evaluation (Test Set: {len(test)} days) ---\n")
    print(f"  RMSE : {rmse:.2f} USD")
    print(f"  MAPE : {mape:.2f}%")

    # ---- Plot: Forecast vs Actual ----
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(train.index, train['price'], color="#2563EB", linewidth=1.2, label="Training Data")
    ax.plot(test.index, test['price'],   color="#16A34A", linewidth=2.0, label="Actual (Test Set)")
    ax.plot(forecast_mean.index, forecast_mean, color="#DC2626", linewidth=2.0,
            linestyle="--", label="SARIMA Forecast")
    ax.fill_between(conf_int.index,
                    conf_int.iloc[:, 0],
                    conf_int.iloc[:, 1],
                    color="#DC2626", alpha=0.15, label="95% Confidence Interval")

    ax.set_title(f"SARIMA Forecast vs Actual  |  RMSE: {rmse:.2f}  |  MAPE: {mape:.2f}%",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = "outputs/plots/05_sarima_forecast.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {output_path}")

    return forecast_mean, conf_int, rmse, mape


# ============================================================
# STAGE 11 — FUTURE FORECAST 
# ============================================================

def forecast_future(df, future_days=30):
    # --------------------------------------------------------
    # Now we retrain SARIMA on the FULL dataset (train + test)
    # so it uses all available information, then forecast
    # future values beyond the end of the dataset.
    #
    # This is the forecast that would be used in a real
    # business setting showing what the model predicts
    # for the next N days into the future.
    # --------------------------------------------------------
    print(f"\n  Fitting final SARIMA model on full dataset for {future_days}-day forecast...")

    final_model = SARIMAX(
        df['price'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 5),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    future_forecast = final_model.get_forecast(steps=future_days)
    future_mean = future_forecast.predicted_mean
    future_conf = future_forecast.conf_int()

    # ---- Plot: Full History + Future Forecast ----
    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(df.index, df['price'], color="#2563EB", linewidth=1.2, label="Historical Data")
    ax.plot(future_mean.index, future_mean, color="#DC2626", linewidth=2.0,
            linestyle="--", label=f"{future_days}-day Forecast")
    ax.fill_between(future_conf.index,
                    future_conf.iloc[:, 0],
                    future_conf.iloc[:, 1],
                    color="#DC2626", alpha=0.15, label="95% Confidence Interval")
    ax.axvline(x=df.index[-1], color="#6B7280", linewidth=1.2, linestyle=":", label="Forecast Start")

    ax.set_title(f"SARIMA Future Forecast — Next {future_days} Days", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = "outputs/plots/06_future_forecast.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {output_path}")

    # Save forecast values to CSV
    future_df = pd.DataFrame({
        "forecast": future_mean,
        "lower_95": future_conf.iloc[:, 0],
        "upper_95": future_conf.iloc[:, 1]
    })
    csv_path = "data/sarima_future_forecast.csv"
    future_df.to_csv(csv_path)
    print(f"  Forecast saved: {csv_path}")

    return future_mean, future_conf


# ============================================================
# STAGE 12 — SUMMARY REPORT
# ============================================================

def print_summary(rmse, mape):
    # --------------------------------------------------------
    # Prints a clean summary of what the pipeline produced.
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("=" * 60)
    print("\n  Dataset        : Apple (AAPL) daily closing prices (2020-2024)")
    print("  Observations   : 1825 days (weekends filled)")
    print("  Model          : SARIMA(1,1,1)(1,1,1,5)")
    print("\n  --- Model Performance (Test Set: last 60 days) ---")
    print(f"  RMSE           : {rmse:.2f} USD")
    print(f"  MAPE           : {mape:.2f}%")
    print("\n  --- Outputs ---")
    print("  outputs/plots/01_raw_series.png")
    print("  outputs/plots/02_decomposition.png")
    print("  outputs/plots/03_moving_averages.png")
    print("  outputs/plots/04_exponential_smoothing.png")
    print("  outputs/plots/05_sarima_forecast.png")
    print("  outputs/plots/06_future_forecast.png")
    print("  data/sarima_future_forecast.csv")
    print("\n" + "=" * 60)


# ============================================================
# PIPELINE RUNNER
# ============================================================
# Each stage is called in order.
# The output of one stage is passed as input to the next.
# ============================================================

def run_pipeline():
    # --------------------------------------------------------
    # No hardcoded CSV path to ensure data is 
    # download directly from Yahoo Finance.
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  LEVEL 3 — TIME SERIES ANALYSIS (APPLE STOCK)")
    print("=" * 60)

    # Stage 1 – Create output folders
    create_folders()

    # Stage 2 – Download data
    df = load_data("AAPL", "2020-01-01", "2024-12-31")
    if df is None:
        return
    summarise_data(df)

    # Stage 3 – Plot raw time series
    print("\n  [Stage 3] Plotting raw time series...")
    plot_raw_series(df)

    # Stage 4 – Check stationarity with ADF test
    print("\n  [Stage 4] Checking stationarity...")
    check_stationarity(df['price'])

    # Stage 5 – Decompose into trend, seasonality, residual
    print("\n  [Stage 5] Decomposing time series...")
    decompose_series(df)

    # Stage 6 – Apply moving average smoothing
    print("\n  [Stage 6] Applying moving averages...")
    apply_moving_average(df)

    # Stage 7 – Apply Holt-Winters exponential smoothing
    print("\n  [Stage 7] Applying exponential smoothing...")
    apply_exponential_smoothing(df)

    # Stage 8 – Split into train and test sets
    print("\n  [Stage 8] Splitting data...")
    train, test = split_data(df, test_days=60)

    # Stage 9 – Fit SARIMA model on training data
    print("\n  [Stage 9] Building SARIMA model...")
    sarima_result = build_sarima_model(train)

    # Stage 10 – Forecast on test set and evaluate accuracy
    print("\n  [Stage 10] Forecasting and evaluating...")
    forecast_mean, conf_int, rmse, mape = forecast_and_evaluate(sarima_result, train, test, df)

    # Stage 11 – Forecast future values beyond the dataset
    print("\n  [Stage 11] Generating future forecast...")
    forecast_future(df, future_days=30)

    # Stage 12 – Print final summary
    print_summary(rmse, mape)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline()