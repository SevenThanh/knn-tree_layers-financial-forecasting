import numpy as np

def calc_ma(ts, period):
    n = len(ts)
    ma = np.zeros(n - period)
    
    for t in range(period, n):
        win = ts[t - period : t]
        ma[t - period] = np.mean(win)
    return ma

def calc_rsi(ts, period=14):
    diff = np.diff(ts)
    n = len(diff)
    rsi = np.zeros(n - period)
    
    for t in range(period, n):
        win = diff[t - period : t]
        gains = np.maximum(win, 0)
        losses = np.abs(np.minimum(win, 0))
        
        avg_g = np.mean(gains)
        avg_l = np.mean(losses)
        
        if avg_l < 1e-10:
            rsi[t - period] = 100.0
        else:
            rs = avg_g / avg_l
            rsi[t - period] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calc_macd(ts, fast=12, slow=26, sig=9):
    def ema(arr, span):
        alpha = 2.0 / (span + 1)
        n = len(arr)
        out = np.zeros(n)
        out[0] = arr[0]
        
        for t in range(1, n):
            out[t] = alpha * arr[t-1] + (1 - alpha) * out[t-1]
        
        return out
    
    min_len = slow + sig
    if len(ts) <= min_len:
        raise ValueError(f"Need at least {min_len} points for MACD")
    
    ema_f = ema(ts, fast)
    ema_s = ema(ts, slow)
    
    macd_line = ema_f - ema_s
    sig_line = ema(macd_line, sig)
    hist = macd_line - sig_line
    
    start = slow
    return macd_line[start:], sig_line[start:], hist[start:]


def calc_volatility(ts, period=20):
    rets = np.diff(ts) / (ts[:-1] + 1e-10)
    n = len(rets)
    vol = np.zeros(n - period)
    
    for t in range(period, n):
        win = rets[t - period : t]
        vol[t - period] = np.std(win, ddof=1)
    
    return vol

def calc_lags(ts, lags):
    max_lag = max(lags)
    n = len(ts)
    out_len = n - max_lag
    
    feats = np.zeros((out_len, len(lags)))
    
    for i, k in enumerate(lags):
        for t in range(max_lag, n):
            feats[t - max_lag, i] = ts[t - k]
    
    return feats


def calc_returns(ts, periods):
    max_p = max(periods)
    n = len(ts)
    out_len = n - max_p - 1
    
    rets = np.zeros((out_len, len(periods)))
    
    for i, p in enumerate(periods):
        for t in range(max_p + 1, n):
            idx_end = t - 1
            idx_start = t - 1 - p
            rets[t - max_p - 1, i] = (ts[idx_end] - ts[idx_start]) / (ts[idx_start] + 1e-10)
    
    return rets