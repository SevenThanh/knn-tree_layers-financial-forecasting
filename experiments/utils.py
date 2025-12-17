import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

# preprocessing series

def clean_daily_series(row):
    # Drop the ID in column V1
    ts = row.iloc[1:]

    # Drop trailing NaNs (uneven lengths)
    ts = ts.dropna().astype(float)

    # Assign daily index (fake but consistent)
    ts.index = pd.date_range(start="2000-01-01", periods=len(ts), freq="D")

    return ts

def build_windows(ts, L, H):
    """
    ts: 1D numpy array
    L: lookback window
    H: forecast horizon
    Returns: X [N,L], Y [N,H]
    """
    X, Y = [], []
    for i in range(len(ts) - L - H + 1):
        X.append(ts[i:i+L])
        Y.append(ts[i+L:i+L+H])
    return np.array(X), np.array(Y)

# evaluation metrics:

def rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def smape(actual, predicted, eps=1e-8):
    numerator = np.abs(actual - predicted)
    denominator = np.abs(actual) + np.abs(predicted) + eps
    return 200.0 * np.mean(numerator / denominator)

def mase(actual, predicted, y_train, m=1, eps=1e-8):
    """
    actual: test series
    predicted: forecasted values
    y_train: full training series
    m: seasonal period
    """
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    y_train = np.asarray(y_train).flatten()

    # enforce proper lag
    m_value = min(m, len(y_train)-1)
    if m_value < 1:
        return np.nan  # cannot compute MASE

    # denominator: mean absolute naive forecast error
    naive_errors = np.abs(y_train[m_value:] - y_train[:-m_value])
    scale = np.mean(naive_errors) + eps

    # numerator: mean absolute forecast error
    mae_forecast = np.mean(np.abs(actual - predicted))

    return mae_forecast / scale

def directional_accuracy(actual, predicted):
    # will not work for 1 day forecasting since need more than 2 values
    if len(actual) < 2:
        return np.nan

    true_diff = np.sign(np.diff(actual))
    pred_diff = np.sign(np.diff(predicted))

    return np.mean(true_diff == pred_diff)

# displaying evalution results:

def print_evaluation_table(rmses, maes, smapes, mases, das):
    summary_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "sMAPE (%)", "MASE", "DA"],
        "Mean": [
            np.mean(rmses),
            np.mean(maes),
            np.mean(smapes),
            np.mean(mases),
            np.nanmean(das)
        ],
        "Median": [
            np.median(rmses),
            np.median(maes),
            np.median(smapes),
            np.median(mases),
            np.nanmedian(das)
        ]
    })
    summary_df[["Mean", "Median"]] = summary_df[["Mean", "Median"]].round(4)
    print(summary_df)