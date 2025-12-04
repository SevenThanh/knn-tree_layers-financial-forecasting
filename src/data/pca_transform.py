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