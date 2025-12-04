import numpy as np
from data_loader import M4Loader, create_synthetic_m4
from rolling_norm import RollingNormalizer
from tech_indicators import FeatureBuilder
from pca_transform import TemporalPCA, temporal_split

class Pipeline:
    def __init__(self, win_sz=30, n_comp=10, train_pct=0.7, val_pct=0.15):
        self.win_sz = win_sz
        self.n_comp = n_comp
        self.train_pct = train_pct
        self.val_pct = val_pct
        
        self.feat_builder = FeatureBuilder()
        self.normalizer = RollingNormalizer(win_sz=win_sz)
        self.pca = TemporalPCA(n_comp=n_comp)
        
        self.feat_names = []
        self.fitted = False
    
    def process_series(self, ts, sid=None):
        feats, self.feat_names = self.feat_builder.build(ts)
        
        norm_feats, start_idx = self.normalizer.transform_multi(feats)
        
        if not self.normalizer.verify_no_lookahead():
            raise RuntimeError(f"Data leakage in series {sid}")
        
        offset = len(ts) - len(feats) + start_idx
        targets = ts[offset:]
        
        min_len = min(len(norm_feats), len(targets))
        norm_feats = norm_feats[:min_len]
        targets = targets[:min_len]
        
        X_tr, X_val, X_te = temporal_split(norm_feats, self.train_pct, self.val_pct)
        y_tr, y_val, y_te = temporal_split(targets, self.train_pct, self.val_pct)
        
        self.pca.fit(X_tr)
        self.fitted = True
        
        X_tr_pca = self.pca.transform(X_tr)
        X_val_pca = self.pca.transform(X_val)
        X_te_pca = self.pca.transform(X_te)
        
        return {
            'id': sid,
            'X_train': X_tr_pca,
            'X_val': X_val_pca,
            'X_test': X_te_pca,
            'y_train': y_tr,
            'y_val': y_val,
            'y_test': y_te,
            'feat_names': self.feat_names,
            'var_exp': self.pca.var_exp,
            'total_var': self.pca.total_var
        }
    
    def process_batch(self, data_dict, min_len=None):
        if min_len is None:
            min_len = self.feat_builder.min_len + self.win_sz + 50
        
        results = []
        failed = []
        
        for sid, s in data_dict.items():
            ts = s['full'] if isinstance(s, dict) else s
            
            if len(ts) < min_len:
                failed.append((sid, "too short"))
                continue
            
            try:
                r = self.process_series(ts, sid)
                results.append(r)
            except Exception as e:
                failed.append((sid, str(e)))
        
        if failed:
            print(f"Failed: {len(failed)} series")
        
        return results
    
    def get_combined(self, results):
        X_tr = np.vstack([r['X_train'] for r in results])
        X_val = np.vstack([r['X_val'] for r in results])
        X_te = np.vstack([r['X_test'] for r in results])
        
        y_tr = np.concatenate([r['y_train'] for r in results])
        y_val = np.concatenate([r['y_val'] for r in results])
        y_te = np.concatenate([r['y_test'] for r in results])
        
        return {
            'X_train': X_tr,
            'X_val': X_val,
            'X_test': X_te,
            'y_train': y_tr,
            'y_val': y_val,
            'y_test': y_te
        }