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

class FeatureBuilder:
    def __init__(self, ma_periods=[7, 30, 90], rsi_p=14, macd_cfg=(12, 26, 9),
                 vol_p=20, lags=[1, 2, 5, 10, 20], ret_periods=[1, 5, 20]):
        self.ma_periods = ma_periods
        self.rsi_p = rsi_p
        self.macd_cfg = macd_cfg
        self.vol_p = vol_p
        self.lags = lags
        self.ret_periods = ret_periods
        
        self.min_len = max(
            max(ma_periods),
            rsi_p + 1,
            macd_cfg[1] + macd_cfg[2],
            vol_p + 1,
            max(lags),
            max(ret_periods) + 1
        ) + 10
        
        self.feat_names = []
    
    def build(self, ts):
        n = len(ts)
        if n < self.min_len:
            raise ValueError(f"Need at least {self.min_len} points, got {n}")
        
        mas = {}
        for p in self.ma_periods:
            mas[p] = calc_ma(ts, p)
        
        rsi = calc_rsi(ts, self.rsi_p)
        macd, macd_sig, macd_hist = calc_macd(ts, *self.macd_cfg)
        vol = calc_volatility(ts, self.vol_p)
        lag_feats = calc_lags(ts, self.lags)
        rets = calc_returns(ts, self.ret_periods)
        
        lens = [len(mas[p]) for p in self.ma_periods]
        lens.extend([len(rsi), len(macd), len(vol), len(lag_feats), len(rets)])
        common_len = min(lens)
        
        self.feat_names = []
        feat_list = []
        
        for p in self.ma_periods:
            feat_list.append(mas[p][-common_len:])
            self.feat_names.append(f'ma_{p}')
        
        feat_list.append(rsi[-common_len:])
        self.feat_names.append('rsi')
        
        feat_list.append(macd[-common_len:])
        self.feat_names.append('macd')
        feat_list.append(macd_sig[-common_len:])
        self.feat_names.append('macd_sig')
        feat_list.append(macd_hist[-common_len:])
        self.feat_names.append('macd_hist')
        
        feat_list.append(vol[-common_len:])
        self.feat_names.append('vol')
        
        lag_arr = lag_feats[-common_len:]
        for i, lag in enumerate(self.lags):
            feat_list.append(lag_arr[:, i])
            self.feat_names.append(f'lag_{lag}')
        
        ret_arr = rets[-common_len:]
        for i, p in enumerate(self.ret_periods):
            feat_list.append(ret_arr[:, i])
            self.feat_names.append(f'ret_{p}')
        
        feats = np.column_stack(feat_list)
        return feats, self.feat_names
    

    
         