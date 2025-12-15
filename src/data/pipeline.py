import numpy as np
from .rolling_norm import RollingNormalizer
from .tech_indicators import FeatureBuilder
from .pca_transform import TemporalPCA, temporal_split
from sklearn.preprocessing import StandardScaler

class Pipeline:
    def __init__(self, win_sz=30, n_comp=10, train_pct=0.7, val_pct=0.15, 
                 use_pca=True, target_mode='norm', horizon=1):
        self.win_sz = win_sz
        self.n_comp = n_comp
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.use_pca = use_pca
        self.target_mode = target_mode
        self.horizon = horizon
        
        self.feat_builder = FeatureBuilder()
        self.normalizer = RollingNormalizer(win_sz=win_sz)
        self.pca = TemporalPCA(n_comp=n_comp)
        self.scaler = StandardScaler()
        
        self.feat_names = []
        self.fitted = False
    
    def process_series(self, ts, sid=None):
        feats, raw_names = self.feat_builder.build(ts)
        norm_feats, start_idx = self.normalizer.transform_multi(feats)
        
        if not self.normalizer.verify_no_lookahead():
            raise RuntimeError(f"Data leakage in series {sid}")
        feat_start_in_ts = len(ts) - len(feats) + start_idx
        
        if self.target_mode == 'returns':
            valid_len = len(ts) - feat_start_in_ts - self.horizon
            if valid_len <= 0: raise ValueError(f"Series too short for horizon {self.horizon}")
            
            p_t = ts[feat_start_in_ts : feat_start_in_ts + valid_len]
            p_fut = ts[feat_start_in_ts + self.horizon : feat_start_in_ts + valid_len + self.horizon]
            targets = (p_fut - p_t) / (p_t + 1e-10)
            norm_feats = norm_feats[:valid_len]
            
        else:
            offset = len(ts) - len(feats) + start_idx
            raw_targets = ts[offset:]
            target_norm = RollingNormalizer(win_sz=self.win_sz)
            targets_normalized, t_start = target_norm.fit_transform(raw_targets)
            
            norm_feats = norm_feats[t_start:]
            min_len = min(len(norm_feats), len(targets_normalized))
            norm_feats = norm_feats[:min_len]
            targets = targets_normalized[:min_len]
        X_tr, X_val, X_te = temporal_split(norm_feats, self.train_pct, self.val_pct)
        y_tr, y_val, y_te = temporal_split(targets, self.train_pct, self.val_pct)
        
        if self.use_pca:
            if not self.fitted:
                self.pca.fit(X_tr)
                self.fitted = True
                self.feat_names = [f'PC{i}' for i in range(self.n_comp)]
            
            X_tr_out = self.pca.transform(X_tr)
            X_val_out = self.pca.transform(X_val)
            X_te_out = self.pca.transform(X_te)
        else:
            if not self.fitted:
                self.scaler.fit(X_tr) 
                self.fitted = True
                self.feat_names = raw_names
            
            X_tr_out = self.scaler.transform(X_tr)
            X_val_out = self.scaler.transform(X_val)
            X_te_out = self.scaler.transform(X_te)
            
        return {
            'id': sid,
            'X_train': X_tr_out, 'X_val': X_val_out, 'X_test': X_te_out,
            'y_train': y_tr, 'y_val': y_val, 'y_test': y_te,
            'feat_names': self.feat_names
        }
    
    def process_batch(self, data_dict, min_len=None):
        if min_len is None:
            min_len = self.feat_builder.min_len + self.win_sz + 50
        
        results = []
        self.fitted = False 
        
        for sid, s in data_dict.items():
            ts = s['full'] if isinstance(s, dict) else s
            if len(ts) < min_len: continue
            
            try:
                results.append(self.process_series(ts, sid))
            except Exception:
                continue
                
        return results
    
    def get_combined(self, results):
        if not results: return None
        return {
            'X_train': np.vstack([r['X_train'] for r in results]),
            'X_val': np.vstack([r['X_val'] for r in results]),
            'X_test': np.vstack([r['X_test'] for r in results]),
            'y_train': np.concatenate([r['y_train'] for r in results]),
            'y_val': np.concatenate([r['y_val'] for r in results]),
            'y_test': np.concatenate([r['y_test'] for r in results])
        }