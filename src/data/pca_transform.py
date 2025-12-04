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