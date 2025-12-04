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