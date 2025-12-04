import numpy as np
from sklearn.decomposition import PCA

class TemporalPCA:
    def __init__(self, n_comp=10, var_thresh=None):
        self.n_comp = n_comp
        self.var_thresh = var_thresh
        self.pca = None
        self.fitted = False
        self.var_exp = None
        self.total_var = None
    
    def fit(self, X_train):
        if self.var_thresh is not None:
            self.pca = PCA(n_components=self.var_thresh, svd_solver='full')
        else:
            self.pca = PCA(n_components=self.n_comp)
        
        self.pca.fit(X_train)
        self.fitted = True
        
        self.var_exp = self.pca.explained_variance_ratio_
        self.total_var = np.sum(self.var_exp)
        
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("PCA not fitted. Call fit() with training data first.")
        return self.pca.transform(X)
    
    def fit_transform(self, X_train):
        self.fit(X_train)
        return self.transform(X_train)
    
    def get_loadings(self):
        if not self.fitted:
            return None
        return self.pca.components_
    
    def inverse(self, X_pca):
        if not self.fitted:
            raise RuntimeError("PCA not fitted.")
        return self.pca.inverse_transform(X_pca)
    
def temporal_split(data, train_pct=0.7, val_pct=0.15):
    n = len(data)
    tr_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))
    
    train = data[:tr_end]
    val = data[tr_end:val_end]
    test = data[val_end:]
    
    return train, val, test

class PCAPipeline:
    def __init__(self, n_comp=10, train_pct=0.7, val_pct=0.15):
        self.pca = TemporalPCA(n_comp=n_comp)
        self.train_pct = train_pct
        self.val_pct = val_pct
    
    def process(self, X, y=None):
        X_tr, X_val, X_te = temporal_split(X, self.train_pct, self.val_pct)
        
        self.pca.fit(X_tr)
        
        X_tr_pca = self.pca.transform(X_tr)
        X_val_pca = self.pca.transform(X_val)
        X_te_pca = self.pca.transform(X_te)
        
        result = {
            'X_train': X_tr_pca,
            'X_val': X_val_pca,
            'X_test': X_te_pca,
            'var_exp': self.pca.var_exp,
            'total_var': self.pca.total_var,
            'n_comp': self.pca.pca.n_components_
        }
        
        if y is not None:
            y_tr, y_val, y_te = temporal_split(y, self.train_pct, self.val_pct)
            result['y_train'] = y_tr
            result['y_val'] = y_val
            result['y_test'] = y_te
        
        return result
