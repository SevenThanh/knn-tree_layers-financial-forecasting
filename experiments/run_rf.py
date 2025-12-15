import argparse
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.data_loader import create_synthetic_m4, M4Loader
from src.data.pipeline import Pipeline
from src.evaluation.metrics import eval_all
from src.models.rf_model import RFTrainer
from src.evaluation.rf_interpretability import RFExplainer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--use_m4', action='store_true')
    p.add_argument('--m4_dir', type=str, default=None)
    p.add_argument('--n_series', type=int, default=50)
    p.add_argument('--category', type=str, default='Finance')
    p.add_argument('--freq', type=str, default='Daily')
    p.add_argument('--horizons', nargs='+', type=int, default=[1, 5, 20])
    p.add_argument('--n_trees', type=int, default=100)
    p.add_argument('--max_depth', type=int, default=10)
    p.add_argument('--min_samples_leaf', type=int, default=5, help="Increase to regularize/smooth predictions")
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def run_horizon(data, horizon, args):
    print(f"\n{'='*40}\nRunning Horizon: {horizon} (Target: Returns)\n{'='*40}")
    pipe = Pipeline(win_sz=30, n_comp=10, use_pca=False, 
                    target_mode='returns', horizon=horizon)
    
    results = pipe.process_batch(data)
    if not results:
        print("No series processed.")
        return
        
    combined = pipe.get_combined(results)
    feat_names = results[0]['feat_names']
    
    print(f"Features: {len(feat_names)} ({feat_names[:3]}...)")
    print(f"Training samples: {combined['X_train'].shape[0]}")
    trainer = RFTrainer(
        n_trees=args.n_trees, 
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf, 
        random_state=args.seed
    )
    
    trainer.fit(combined['X_train'], combined['y_train'], 
                combined['X_val'], combined['y_val'], feat_names=feat_names, verbose=True)
    
    y_pred, unc = trainer.predict(combined['X_test'], return_uncertainty=True)
    y_prev = np.zeros_like(combined['y_test'])
    
    metrics = eval_all(combined['y_test'], y_pred, combined['y_train'], y_prev)
    print(f"SMAPE: {metrics['smape']:.2f}")
    print(f"MASE:  {metrics['mase']:.4f}")
    print(f"DirAcc: {metrics['dir_acc']:.1f}%")
    
    print("\n--- Interpretability Snapshot ---")
    print("Top Rules:")
    rules = trainer.extract_rules(max_rules=3)
    for r in rules:
        print(f"  {r['rule_str'][:120]}")

def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    if args.use_m4:
        loader = M4Loader(args.m4_dir)
        # note: limiting to 100 series for speed during tuning
        data = loader.load_category(args.category, freq=args.freq, max_n=args.n_series)
    else:
        data = create_synthetic_m4(n_series=args.n_series)
        
    for h in args.horizons:
        run_horizon(data, h, args)

if __name__ == "__main__":
    main()