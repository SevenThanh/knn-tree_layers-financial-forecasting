import numpy as np
from collections import defaultdict

class RFExplainer:
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = trainer.model
        
    def summarize_importance(self, top_k=10):
        imp_data = self.model.get_feature_importance(sort=True)
        ranking = []
        for i in range(min(top_k, len(imp_data['names']))):
            ranking.append({
                'rank': i + 1, 'feature': imp_data['names'][i],
                'importance': imp_data['importance'][i],
                'stability': imp_data['stability'][i]
            })
        cv = imp_data['stability'] / (imp_data['importance'] + 1e-10)
        stable_mask = cv < np.median(cv)
        stable = [imp_data['names'][i] for i in range(len(cv)) 
                 if stable_mask[i] and imp_data['importance'][i] > 0.01]
        unstable = [imp_data['names'][i] for i in range(len(cv))
                   if not stable_mask[i] and imp_data['importance'][i] > 0.01]
        return {'ranking': ranking, 'stable_features': stable[:5], 'unstable_features': unstable[:5]}
    
    def analyze_temporal_drift(self):
        temporal_imp = self.model.get_temporal_importance()
        if len(temporal_imp) < 2:
            return {'drift': None, 'message': 'Not enough temporal windows'}
        n_features = len(temporal_imp[0]['importance'])
        trajectories = defaultdict(list)
        for window in temporal_imp:
            for i in range(n_features):
                trajectories[self.model.feat_names[i]].append(window['importance'][i])
        drift_analysis = []
        for feat, values in trajectories.items():
            if len(values) >= 2:
                slope = (values[-1] - values[0]) / len(values)
                drift_analysis.append({
                    'feature': feat, 'slope': slope,
                    'start_imp': values[0], 'end_imp': values[-1]
                })
        drift_analysis.sort(key=lambda x: abs(x['slope']), reverse=True)
        emerging = [d for d in drift_analysis if d['slope'] > 0.01][:3]
        fading = [d for d in drift_analysis if d['slope'] < -0.01][:3]
        stable = [d for d in drift_analysis if abs(d['slope']) <= 0.01 and d['end_imp'] > 0.05][:3]
        return {'emerging': emerging, 'fading': fading, 'stable': stable}
    
    def extract_contrastive_rules(self, X, y, n_rules=5, threshold_pct=25):
        preds = self.model.predict(X)
        high_thresh = np.percentile(preds, 100 - threshold_pct)
        low_thresh = np.percentile(preds, threshold_pct)
        high_mask = preds >= high_thresh
        low_mask = preds <= low_thresh
        all_rules = self.model.extract_rules(max_rules=50, min_samples=20)
        contrastive = []
        for rule in all_rules:
            satisfies = self._check_rule(X, rule['conditions'])
            if satisfies.sum() < 10:
                continue
            high_rate = (satisfies & high_mask).sum() / max(satisfies.sum(), 1)
            low_rate = (satisfies & low_mask).sum() / max(satisfies.sum(), 1)
            disc = abs(high_rate - low_rate)
            contrastive.append({
                'rule': rule['rule_str'], 'discrimination': disc,
                'direction': 'HIGH' if high_rate > low_rate else 'LOW'
            })
        contrastive.sort(key=lambda r: r['discrimination'], reverse=True)
        return contrastive[:n_rules]
    
    def _check_rule(self, X, conditions):
        satisfies = np.ones(X.shape[0], dtype=bool)
        for fname, op, thresh in conditions:
            try:
                fidx = self.model.feat_names.index(fname)
            except ValueError:
                continue
            if op == '<=':
                satisfies &= (X[:, fidx] <= thresh)
            else:
                satisfies &= (X[:, fidx] > thresh)
        return satisfies
    
    def compute_prediction_intervals(self, X, confidence=0.9):
        tree_preds = np.array([t.predict(X) for t in self.model.forest.estimators_])
        alpha = (1 - confidence) / 2
        return {
            'prediction': np.mean(tree_preds, axis=0),
            'std': np.std(tree_preds, axis=0),
            'lower': np.percentile(tree_preds, alpha * 100, axis=0),
            'upper': np.percentile(tree_preds, (1 - alpha) * 100, axis=0)
        }
    
    def generate_report(self, X_test, y_test):
        report = {}
        report['importance'] = self.summarize_importance(top_k=10)
        report['temporal_drift'] = self.analyze_temporal_drift()
        report['contrastive_rules'] = self.extract_contrastive_rules(X_test, y_test)
        intervals = self.compute_prediction_intervals(X_test)
        in_interval = (y_test >= intervals['lower']) & (y_test <= intervals['upper'])
        report['interval_coverage'] = {
            'target': 0.9, 'actual': np.mean(in_interval),
            'mean_width': np.mean(intervals['upper'] - intervals['lower'])
        }
        return report
    
    def print_report(self, report):
        print("=" * 50)
        print("RF INTERPRETABILITY REPORT")
        print("=" * 50)
        print("\n--- FEATURE IMPORTANCE ---")
        for item in report['importance']['ranking'][:5]:
            stab = "stable" if item['stability'] < 0.05 else "variable"
            print(f"  {item['rank']}. {item['feature']}: {item['importance']:.4f} ({stab})")
        print("\n--- TEMPORAL DRIFT ---")
        drift = report['temporal_drift']
        if drift.get('emerging'):
            print("  Emerging:")
            for f in drift['emerging']:
                print(f"    {f['feature']}: {f['start_imp']:.3f} -> {f['end_imp']:.3f}")
        if drift.get('fading'):
            print("  Fading:")
            for f in drift['fading']:
                print(f"    {f['feature']}: {f['start_imp']:.3f} -> {f['end_imp']:.3f}")
        print("\n--- CONTRASTIVE RULES ---")
        for i, rule in enumerate(report['contrastive_rules'][:3]):
            print(f"  {i+1}. [{rule['direction']}] disc={rule['discrimination']:.3f}")
        print("\n--- PREDICTION INTERVALS ---")
        iv = report['interval_coverage']
        print(f"  Target: {iv['target']*100:.0f}%, Actual: {iv['actual']*100:.1f}%")
        print("=" * 50)