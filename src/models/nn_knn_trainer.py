import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.nnknn import NNKNN


class NNKNNTrainer:
    """
    Training wrapper for NN-kNN model.
    
    Handles:
    - Converting numpy arrays to tensors
    - Storing case base (training examples)
    - Batched training loop
    - Prediction with stored cases
    """
    
    def __init__(self, n_feat, n_cases, n_out=1, shared_wts=False, 
                 lr=0.01, device=None):
        """
        Args:
            n_feat: number of input features
            n_cases: number of training cases to store
            n_out: output dimension (1 for single horizon forecast)
            shared_wts: if True, share weights across cases
            lr: learning rate
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.n_feat = n_feat
        self.n_cases = n_cases
        self.n_out = n_out
        self.lr = lr
        
        self.model = NNKNN(
            num_features=n_feat,
            num_cases=n_cases,
            shared_weights=shared_wts
        ).to(self.device)

        
        self.cases = None     
        self.targets = None   
        
        self.optim = optim.Adam(self.model.parameters(), lr=lr)
        
        self.loss_fn = nn.MSELoss()
        
        self.history = {'train_loss': [], 'val_loss': []}
    
    def set_cases(self, X, y):
        """
        Store training examples as the case base.
        
        Args:
            X: np.array [N, D] features
            y: np.array [N] or [N, C] targets
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        MAX_CASES = 20000
        if X.shape[0] > MAX_CASES:
            print(f"⚠️  Subsampling {X.shape[0]} → {MAX_CASES} cases")
            idx = np.random.choice(X.shape[0], MAX_CASES, replace=False)
            X = X[idx]
            y = y[idx]
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        self.cases = torch.tensor(X, device=self.device)
        self.targets = torch.tensor(y, device=self.device)
        
        if self.cases.shape[0] != self.n_cases:
            print(f"Warning: n_cases mismatch. Expected {self.n_cases}, got {self.cases.shape[0]}")
            print("Reinitializing model with correct n_cases...")
            self.n_cases = self.cases.shape[0]
            self.model = NNKNN(
                num_features=self.n_feat,
                num_cases=self.n_cases,
                shared_weights=self.model.feature_distance.shared_weights
            ).to(self.device)
            self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
    
    def _to_tensor(self, arr):
        """Convert numpy array to tensor on correct device."""
        arr = np.asarray(arr, dtype=np.float32)
        return torch.tensor(arr, device=self.device)
    
    def train_epoch(self, X_train, y_train, batch_sz=32):
        """
        One training epoch.
        
        Uses X_train as queries, compares against stored cases.
        
        Args:
            X_train: np.array [M, D] training queries
            y_train: np.array [M] or [M, C] query targets
            batch_sz: batch size
        
        Returns:
            float: average loss for epoch
        """
        self.model.train()
        
        X = self._to_tensor(X_train)
        y = self._to_tensor(y_train)
        if y.ndim == 1:
            y = y.unsqueeze(1)
        
        n_samples = X.shape[0]
        indices = torch.randperm(n_samples)
        
        total_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_sz):
            batch_idx = indices[i:i+batch_sz]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            self.optim.zero_grad()
            
            y_pred, activations, distances = self.model(
                X_batch, self.cases, self.targets, q_indices=batch_idx
            )
            
            loss = self.loss_fn(y_pred, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optim.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def eval_loss(self, X_val, y_val, batch_sz=64):
        """
        Compute loss on validation set.
        
        Args:
            X_val: np.array [M, D]
            y_val: np.array [M] or [M, C]
            batch_sz: batch size
        
        Returns:
            float: average loss
        """
        self.model.eval()
        
        X = self._to_tensor(X_val)
        y = self._to_tensor(y_val)
        if y.ndim == 1:
            y = y.unsqueeze(1)
        
        n_samples = X.shape[0]
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_sz):
                X_batch = X[i:i+batch_sz]
                y_batch = y[i:i+batch_sz]
                
                y_pred, _, _ = self.model(X_batch, self.cases, self.targets)
                loss = self.loss_fn(y_pred, y_batch)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=100, batch_sz=32, patience=10, verbose=True):
        """
        Full training loop with early stopping.
        
        Args:
            X_train: training features
            y_train: training targets
            X_val: validation features (optional)
            y_val: validation targets (optional)
            epochs: max epochs
            batch_sz: batch size
            patience: early stopping patience
            verbose: print progress
        
        Returns:
            dict: training history
        """
        self.set_cases(X_train, y_train)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for ep in range(epochs):
            train_loss = self.train_epoch(X_train, y_train, batch_sz)
            self.history['train_loss'].append(train_loss)
            
            if X_val is not None and y_val is not None:
                val_loss = self.eval_loss(X_val, y_val, batch_sz)
                self.history['val_loss'].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if verbose and (ep + 1) % 10 == 0:
                    print(f"Epoch {ep+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {ep+1}")
                    break
            else:
                if verbose and (ep + 1) % 10 == 0:
                    print(f"Epoch {ep+1}/{epochs} | Train: {train_loss:.4f}")
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return self.history
    
    def predict(self, X, return_info=False):
        """
        Make predictions for new data.
        
        Args:
            X: np.array [M, D] input features
            return_info: if True, also return activations and distances
        
        Returns:
            np.array [M] or [M, C]: predictions
            (optional) activations [M, N], distances [M, N, D]
        """
        self.model.eval()
        
        X_t = self._to_tensor(X)
        
        with torch.no_grad():
            y_pred, acts, dists = self.model(X_t, self.cases, self.targets)
        
        y_pred = y_pred.cpu().numpy()
        
        if y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
        
        if return_info:
            acts = acts.cpu().numpy()
            dists = dists.cpu().numpy()
            return y_pred, acts, dists
        
        return y_pred
    
    def save(self, path):
        """Save model and case base."""
        torch.save({
            'model_state': self.model.state_dict(),
            'cases': self.cases.cpu(),
            'targets': self.targets.cpu(),
            'n_feat': self.n_feat,
            'n_cases': self.n_cases,
            'n_out': self.n_out,
            'history': self.history
        }, path)
    
    def load(self, path):
        """Load model and case base."""
        ckpt = torch.load(path, map_location=self.device)
        
        self.n_feat = ckpt['n_feat']
        self.n_cases = ckpt['n_cases']
        self.n_out = ckpt['n_out']
        self.history = ckpt['history']
        
        self.model = NNKNN(
            num_features=self.n_feat,
            num_cases=self.n_cases
        ).to(self.device)
        
        self.model.load_state_dict(ckpt['model_state'])
        self.cases = ckpt['cases'].to(self.device)
        self.targets = ckpt['targets'].to(self.device)

