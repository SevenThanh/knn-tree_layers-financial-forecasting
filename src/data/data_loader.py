import numpy as np
import pandas as pd
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent / "m4_data"

class M4Loader:
    def __init__(self, data_dir=None):
        if data_dir is None:
            self.data_dir = BASE_DIR
        else:
            self.data_dir = Path(data_dir)  
        self.info = None
        self.cache = {}
        
    def _load_info(self):
        if self.info is not None:
            return self.info

        fpath = self.data_dir / "m4_info.csv"
        
        if not fpath.exists():
            print(f"Error: {fpath} not found.")
            return None
            
        self.info = pd.read_csv(fpath)
        return self.info
    
    def _load_freq_data(self, freq, split="train"):
        key = f"{freq}_{split}"
        if key in self.cache:
            return self.cache[key]
        
        fname = f"{freq}-{split}.csv"
        fpath = self.data_dir / fname
        
        if not fpath.exists():
            print(f"Error: {fpath} not found.")
            return None
            
        df = pd.read_csv(fpath)
        self.cache[key] = df
        return df
    
    def get_series_ids(self, freq=None, cat=None):
        info = self._load_info()
        if info is None:
            return []
        
        mask = pd.Series([True] * len(info))
        if freq:
            mask &= info['SP'] == freq
        if cat:
            mask &= info['category'] == cat
            
        return info.loc[mask, 'M4id'].tolist()
    
    def load_series(self, sid):
        info = self._load_info()
        if info is None:
            return None
            
        row = info[info['M4id'] == sid]
        if len(row) == 0:
            return None
        row = row.iloc[0]
        
        freq = row['SP']
        df_train = self._load_freq_data(freq, "train")
        df_test = self._load_freq_data(freq, "test")
        
        if df_train is None:
            return None
            
        train_row = df_train[df_train['V1'] == sid]
        if len(train_row) == 0:
            return None
            
        # Extract values (skipping V1 column)
        train_vals = train_row.iloc[0, 1:].dropna().values.astype(np.float64)
        
        test_vals = np.array([])
        if df_test is not None:
            test_row = df_test[df_test['V1'] == sid]
            if len(test_row) > 0:
                test_vals = test_row.iloc[0, 1:].dropna().values.astype(np.float64)
        
        full = np.concatenate([train_vals, test_vals]) if len(test_vals) > 0 else train_vals
        
        return {
            'id': sid,
            'freq': freq,
            'cat': row['category'],
            'horizon': int(row['Horizon']),
            'train': train_vals,
            'test': test_vals,
            'full': full,
            'n': len(full)
        }
    
    def load_batch(self, sids, min_len=100):
        results = {}
        for sid in sids:
            s = self.load_series(sid)
            if s is None:
                continue
            if s['n'] < min_len:
                continue
            if not self._validate(s['full']):
                continue
            results[sid] = s
        return results
    
    def load_category(self, cat, freq=None, min_len=100, max_n=None):
        sids = self.get_series_ids(freq=freq, cat=cat)
        if max_n:
            sids = sids[:max_n]
        return self.load_batch(sids, min_len=min_len)
    
    def _validate(self, ts):
        if ts is None or len(ts) == 0:
            return False
        if not np.isfinite(ts).all():
            return False
        return True
    
    def fill_missing(self, ts, method='ffill'):
        ts = ts.copy()
        mask = ~np.isfinite(ts)
        if not mask.any():
            return ts
            
        if method == 'ffill':
            for i in range(1, len(ts)):
                if mask[i]:
                    ts[i] = ts[i-1]
        elif method == 'interp':
            valid_idx = np.where(~mask)[0]
            if len(valid_idx) < 2:
                return ts
            ts[mask] = np.interp(
                np.where(mask)[0],
                valid_idx,
                ts[valid_idx]
            )
        return ts
    def normalize_rolling(self, series, window_size=30):
        s = pd.Series(series)
        rolling_mean = s.shift(1).rolling(window=window_size, min_periods=window_size).mean()
        rolling_std = s.shift(1).rolling(window=window_size, min_periods=window_size).std()
        
        rolling_mean = rolling_mean.fillna(method='bfill').fillna(0)
        rolling_std = rolling_std.fillna(method='bfill').fillna(1)
        rolling_std = rolling_std.replace(0, 1e-8)
        
        normalized = (series - rolling_mean.values) / rolling_std.values
        return normalized


