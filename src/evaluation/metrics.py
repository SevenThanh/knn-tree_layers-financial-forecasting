import numpy as np

def calc_smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error.
    
    Formula: (100/n) * sum(|pred - true| / ((|pred| + |true|) / 2))
    
    Handles zero values gracefully. Standard M4 Competition metric.
    
    Args:
        y_true: np.array of actual values
        y_pred: np.array of predicted values
    
    Returns:
        float: SMAPE score (0-200 range, lower is better)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 1e-10
    diff = np.abs(y_pred - y_true)
    
    smape_vals = np.zeros_like(diff)
    smape_vals[mask] = diff[mask] / denom[mask]
    
    return 100.0 * np.mean(smape_vals)


def calc_mase(y_true, y_pred, y_train):
    """
    Mean Absolute Scaled Error.
    
    Formula: MAE(forecast) / MAE(naive forecast on training)
    
    Naive forecast = predict previous value (random walk).
    MASE < 1.0 means model beats naive baseline.
    
    Args:
        y_true: np.array of actual test values
        y_pred: np.array of predicted values
        y_train: np.array of training values (for naive MAE)
    
    Returns:
        float: MASE score (lower is better, <1 beats naive)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_train = np.asarray(y_train).flatten()
    
    mae_pred = np.mean(np.abs(y_pred - y_true))
    
    if len(y_train) < 2:
        return mae_pred
    
    naive_errors = np.abs(y_train[1:] - y_train[:-1])
    mae_naive = np.mean(naive_errors)
    
    if mae_naive < 1e-10:
        return mae_pred
    
    return mae_pred / mae_naive


def calc_dir_acc(y_true, y_pred, y_prev):
    """
    Directional Accuracy.
    
    Measures % of times model correctly predicts direction (up/down).
    Critical for trading applications.
    
    Args:
        y_true: np.array of actual values
        y_pred: np.array of predicted values  
        y_prev: np.array of previous values (to compute direction)
    
    Returns:
        float: Directional accuracy (0-100, higher is better)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_prev = np.asarray(y_prev).flatten()
    
    actual_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    
    matches = (actual_dir == pred_dir)
    
    return 100.0 * np.mean(matches)


def eval_all(y_true, y_pred, y_train, y_prev=None):
    """
    Compute all metrics at once.
    
    Args:
        y_true: actual test values
        y_pred: predicted values
        y_train: training values (for MASE)
        y_prev: previous values for directional accuracy (optional)
    
    Returns:
        dict with 'smape', 'mase', 'dir_acc' keys
    """
    results = {
        'smape': calc_smape(y_true, y_pred),
        'mase': calc_mase(y_true, y_pred, y_train)
    }
    
    if y_prev is not None:
        results['dir_acc'] = calc_dir_acc(y_true, y_pred, y_prev)
    else:
        results['dir_acc'] = None
    
    return results


if __name__ == "__main__":
    np.random.seed(42)
    
    train = np.cumsum(np.random.randn(100)) + 100
    true = np.cumsum(np.random.randn(20)) + train[-1]
    pred = true + np.random.randn(20) * 0.5 
    prev = np.roll(true, 1)
    prev[0] = train[-1]
    
    print("SMAPE:", calc_smape(true, pred))
    print("MASE:", calc_mase(true, pred, train))
    print("Dir Acc:", calc_dir_acc(true, pred, prev))
    
    all_metrics = eval_all(true, pred, train, prev)
    print("All:", all_metrics)