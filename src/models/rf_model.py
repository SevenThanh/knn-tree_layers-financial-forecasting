import numpy as np
from sklearn.ensemble import RandomForestRegressor

class TemporalRandomForest:
    def __init__(self, n_trees=100, max_depth=10, min_samples_leaf=5,
                 n_temp_splits=3, random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_temp_splits = n_temp_splits
        self.random_state = random_state
        self.forest = None
        self.temporal_forests = []
        self.feat_importance = None
        self.feat_importance_std = None
        self.feat_names = None
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y, feat_names=None):
        np.random.seed(self.random_state)
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.feat_names = feat_names if feat_names else [f'f{i}' for i in range(X.shape[1])]
        n_samples = X.shape[0]
        
        self.forest = RandomForestRegressor(
            n_estimators=self.n_trees, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf, random_state=self.random_state,
            n_jobs=-1, bootstrap=True, oob_score=True
        )
        self.forest.fit(X, y)
        
        self.temporal_forests = []
        window_size = n_samples // self.n_temp_splits
        trees_per_window = max(10, self.n_trees // self.n_temp_splits)
        
        for i in range(self.n_temp_splits):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, n_samples)
            if end_idx - start_idx < 50:
                continue
            temp_forest = RandomForestRegressor(
                n_estimators=trees_per_window, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + i, n_jobs=-1
            )
            temp_forest.fit(X[start_idx:end_idx], y[start_idx:end_idx])
            self.temporal_forests.append({
                'forest': temp_forest, 'start': start_idx,
                'end': end_idx, 'n_samples': end_idx - start_idx
            })
        
        self._compute_importance_stability()
        return self
    
    def _compute_importance_stability(self):
        n_features = self.X_train.shape[1]
        tree_importances = np.zeros((self.n_trees, n_features))
        for i, tree in enumerate(self.forest.estimators_):
            tree_importances[i] = tree.feature_importances_
        self.feat_importance = np.mean(tree_importances, axis=0)
        self.feat_importance_std = np.std(tree_importances, axis=0)
        
    def predict(self, X, return_uncertainty=False):
        if return_uncertainty:
            tree_preds = np.array([tree.predict(X) for tree in self.forest.estimators_])
            return np.mean(tree_preds, axis=0), np.std(tree_preds, axis=0)
        return self.forest.predict(X)
    
    def get_feature_importance(self, sort=True):
        if sort:
            idx = np.argsort(-self.feat_importance)
        else:
            idx = np.arange(len(self.feat_importance))
        return {
            'importance': self.feat_importance[idx],
            'stability': self.feat_importance_std[idx],
            'names': [self.feat_names[i] for i in idx],
            'rank': idx
        }
    
    def get_temporal_importance(self):
        results = []
        for i, tf in enumerate(self.temporal_forests):
            imp = tf['forest'].feature_importances_
            results.append({
                'window': i, 'start': tf['start'], 'end': tf['end'],
                'importance': imp,
                'top_features': [self.feat_names[j] for j in np.argsort(-imp)[:5]]
            })
        return results
    
    def get_oob_score(self):
        return getattr(self.forest, 'oob_score_', None)
    
    def extract_rules(self, max_rules=10, min_samples=50):
        rules = []
        for tree_idx, tree in enumerate(self.forest.estimators_[:max_rules]):
            rules.extend(self._extract_tree_rules(tree.tree_, tree_idx, min_samples))
        rules.sort(key=lambda r: abs(r['prediction']), reverse=True)
        return rules[:max_rules]
    
    def _extract_tree_rules(self, tree, tree_idx, min_samples):
        rules = []
        def traverse(node_id, conditions):
            if tree.children_left[node_id] == -1:
                n = tree.n_node_samples[node_id]
                if n >= min_samples:
                    pred = tree.value[node_id][0, 0]
                    rules.append({
                        'tree_idx': tree_idx, 'conditions': conditions.copy(),
                        'prediction': pred, 'n_samples': n,
                        'rule_str': self._format_rule(conditions, pred)
                    })
                return
            feat_idx = tree.feature[node_id]
            thresh = tree.threshold[node_id]
            fname = self.feat_names[feat_idx]
            traverse(tree.children_left[node_id], conditions + [(fname, '<=', thresh)])
            traverse(tree.children_right[node_id], conditions + [(fname, '>', thresh)])
        traverse(0, [])
        return rules
    
    def _format_rule(self, conditions, prediction):
        if not conditions:
            return f"DEFAULT -> {prediction:.4f}"
        parts = [f"{f} {op} {t:.3f}" for f, op, t in conditions]
        return f"IF {' AND '.join(parts)} THEN {prediction:.4f}"
    
    def explain_prediction(self, x, k_rules=3):
        x = np.asarray(x).reshape(1, -1)
        pred, uncertainty = self.predict(x, return_uncertainty=True)
        paths = []
        for tree_idx, tree in enumerate(self.forest.estimators_[:k_rules * 2]):
            node_ids = tree.decision_path(x).indices
            path_info = []
            for nid in node_ids:
                if tree.tree_.children_left[nid] != -1:
                    fidx = tree.tree_.feature[nid]
                    thresh = tree.tree_.threshold[nid]
                    val = x[0, fidx]
                    path_info.append({
                        'feature': self.feat_names[fidx], 'threshold': thresh,
                        'value': val, 'direction': '<=' if val <= thresh else '>'
                    })
            paths.append({'tree_idx': tree_idx, 'prediction': tree.predict(x)[0], 'path': path_info})
        paths.sort(key=lambda p: abs(p['prediction'] - pred[0]))
        return {
            'prediction': pred[0], 'uncertainty': uncertainty[0],
            'decision_paths': paths[:k_rules]
        }

class RFTrainer:
    def __init__(self, n_trees=100, max_depth=10, min_samples_leaf=5,
                 n_temp_splits=3, random_state=42):
        self.model = TemporalRandomForest(
            n_trees=n_trees, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_temp_splits=n_temp_splits, random_state=random_state
        )
        self.history = {'train_loss': [], 'val_loss': []}
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, feat_names=None, verbose=True):
        if verbose:
            print(f"Training RF: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        self.model.fit(X_train, y_train, feat_names)
        train_pred = self.model.predict(X_train)
        train_mse = np.mean((train_pred - y_train) ** 2)
        self.history['train_loss'].append(train_mse)
        if verbose:
            print(f"Train MSE: {train_mse:.4f}")
            oob = self.model.get_oob_score()
            if oob:
                print(f"OOB R2: {oob:.4f}")
        if X_val is not None:
            val_pred = self.model.predict(X_val)
            val_mse = np.mean((val_pred - y_val) ** 2)
            self.history['val_loss'].append(val_mse)
            if verbose:
                print(f"Val MSE: {val_mse:.4f}")
        return self.history
    
    def predict(self, X, return_uncertainty=False):
        return self.model.predict(X, return_uncertainty)
    
    def get_feature_importance(self):
        return self.model.get_feature_importance()
    
    def get_temporal_importance(self):
        return self.model.get_temporal_importance()
    
    def extract_rules(self, max_rules=10):
        return self.model.extract_rules(max_rules)
    
    def explain(self, x):
        return self.model.explain_prediction(x)