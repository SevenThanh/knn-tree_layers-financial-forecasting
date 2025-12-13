import numpy as np

class NNKNNExplainer:
    """
    Extract interpretable explanations from NN-kNN predictions.
    
    For each prediction, identifies which historical cases (neighbors)
    most influenced the forecast and provides human-readable explanations.
    """
    
    def __init__(self, trainer, k=5):
        """
        Args:
            trainer: trained NNKNNTrainer instance
            k: number of top neighbors to retrieve
        """
        self.trainer = trainer
        self.k = k
        
        self.cases = trainer.cases.cpu().numpy()
        self.targets = trainer.targets.cpu().numpy().flatten()
    
    def get_top_neighbors(self, activations, k=None):
        """
        Get indices of top-k activated cases for each query.
        
        Args:
            activations: [M, N] activation scores from model
            k: number of neighbors (uses self.k if None)
        
        Returns:
            indices: [M, k] indices of top neighbors
            scores: [M, k] activation scores of top neighbors
        """
        if k is None:
            k = self.k
        
        sorted_idx = np.argsort(-activations, axis=1)
        top_idx = sorted_idx[:, :k]
        
        m = activations.shape[0]
        top_scores = np.array([
            activations[i, top_idx[i]] for i in range(m)
        ])
        
        return top_idx, top_scores
    
    def explain_prediction(self, query_idx, X_query, y_pred, activations, 
                           timestamps=None, series_ids=None):
        """
        Generate explanation for a single prediction.
        
        Args:
            query_idx: index of query in batch
            X_query: [D] query features
            y_pred: predicted value
            activations: [N] activations for this query
            timestamps: optional list of timestamps for cases
            series_ids: optional list of series IDs for cases
        
        Returns:
            dict with explanation details
        """
        top_idx = np.argsort(-activations)[:self.k]
        top_scores = activations[top_idx]
        
        total_act = np.sum(activations)
        if total_act > 1e-10:
            contrib_pct = (top_scores / total_act) * 100
        else:
            contrib_pct = np.zeros(self.k)
        
        neighbors = []
        for i, idx in enumerate(top_idx):
            nb_info = {
                'case_idx': int(idx),
                'activation': float(top_scores[i]),
                'contribution_pct': float(contrib_pct[i]),
                'target': float(self.targets[idx]),
                'features': self.cases[idx]
            }
            
            if timestamps is not None and idx < len(timestamps):
                nb_info['timestamp'] = timestamps[idx]
            if series_ids is not None and idx < len(series_ids):
                nb_info['series_id'] = series_ids[idx]
            
            neighbors.append(nb_info)
        
        nb_targets = np.array([nb['target'] for nb in neighbors])
        target_std = np.std(nb_targets)
        target_mean = np.mean(nb_targets)
        
        return {
            'prediction': float(y_pred),
            'neighbors': neighbors,
            'neighbor_target_mean': float(target_mean),
            'neighbor_target_std': float(target_std),
            'total_activation': float(total_act)
        }
    
    def explain_batch(self, X, timestamps=None, series_ids=None):
        """
        Generate explanations for a batch of predictions.
        
        Args:
            X: [M, D] input features
            timestamps: optional timestamps for stored cases
            series_ids: optional series IDs for stored cases
        
        Returns:
            list of explanation dicts
        """
        y_pred, acts, _ = self.trainer.predict(X, return_info=True)
        
        explanations = []
        for i in range(len(X)):
            exp = self.explain_prediction(
                query_idx=i,
                X_query=X[i],
                y_pred=y_pred[i] if y_pred.ndim == 1 else y_pred[i, 0],
                activations=acts[i],
                timestamps=timestamps,
                series_ids=series_ids
            )
            explanations.append(exp)
        
        return explanations
    
    def format_explanation(self, exp):
        lines = []
        lines.append(f"Prediction: {exp['prediction']:.4f}")
        lines.append(f"Based on {len(exp['neighbors'])} nearest neighbors:")
        
        for i, nb in enumerate(exp['neighbors']):
            lines.append(f"  {i+1}. Case {nb['case_idx']} | Target: {nb['target']:.4f} | Contrib: {nb['contribution_pct']:.1f}%")
        
        lines.append(f"Neighbor mean: {exp['neighbor_target_mean']:.4f}, std: {exp['neighbor_target_std']:.4f}")
        return "\n".join(lines)
        
    def compute_consistency_score(self, explanations):
        stds = [exp['neighbor_target_std'] for exp in explanations]
        return {
            'mean_neighbor_std': float(np.mean(stds)),
            'median_neighbor_std': float(np.median(stds)),
            'max_neighbor_std': float(np.max(stds)),
            'min_neighbor_std': float(np.min(stds))
        }