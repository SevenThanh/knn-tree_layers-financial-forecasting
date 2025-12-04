import numpy as np

class RollingNormalizer:
    def __init__(self, win_sz=30, eps=1e-8, def_std=0.01, use_expand=False):
        self.win_sz = win_sz
        self.eps = eps
        self.def_std = def_std
        self.use_expand = use_expand
        self.max_ts_used = []
        self.stats = []

    def fit_transform(self, ts):
        n = len(ts)
        self.max_ts_used = []
        self.stats = []
        
        if n <= self.win_sz:
            raise ValueError(f"Series len {n} must be > win_sz {self.win_sz}")
        
        start = 0 if self.use_expand else self.win_sz
        out_len = n - start
        norm = np.zeros(out_len)
        
        for t in range(start, n):
            if self.use_expand and t < self.win_sz:
                w_start = 0
                w_end = t
            else:
                w_start = t - self.win_sz
                w_end = t
            
            win = ts[w_start:w_end]
            
            mu = np.mean(win)
            std = np.std(win, ddof=1)
            
            if std < self.eps:
                std = self.def_std
            norm[t - start] = (ts[t] - mu) / std
            self.max_ts_used.append(w_end - 1)
            self.stats.append((mu, std))
        
        return norm, start

    def transform_multi(self, X):
        n_t, n_f = X.shape
        start = 0 if self.use_expand else self.win_sz
        out_len = n_t - start
        
        norm = np.zeros((out_len, n_f))
        
        for f in range(n_f):
            col, _ = self.fit_transform(X[:, f])
            norm[:, f] = col
            
        return norm, start

    def verify_no_lookahead(self):
        start = 0 if self.use_expand else self.win_sz
        
        for i, max_used in enumerate(self.max_ts_used):
            t = i + start
            if max_used >= t:
                return False
        return True
    
    def get_stats_at(self, idx):
        if idx < 0 or idx >= len(self.stats):
            return None
        return self.stats[idx]
    
