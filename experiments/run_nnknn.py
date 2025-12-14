"""
Experiment script: Train and evaluate NN-kNN on M4 data.

Usage:
    python run_nnknn.py                    # synthetic data test
    python run_nnknn.py --use_m4           # real M4 data
    python run_nnknn.py --n_series 100     # limit series count
"""

import argparse
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.data_loader import create_synthetic_m4
from src.data.pipeline import Pipeline
from src.evaluation.metrics import eval_all


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--use_m4', action='store_true', 
                   help='Use real M4 data instead of synthetic')
    p.add_argument('--n_series', type=int, default=50,
                   help='Number of series to process')
    p.add_argument('--category', type=str, default='Finance',
                   help='M4 category (Finance, Macro, etc)')
    p.add_argument('--epochs', type=int, default=50,
                   help='Training epochs')
    p.add_argument('--batch_sz', type=int, default=32,
                   help='Batch size')
    p.add_argument('--lr', type=float, default=0.01,
                   help='Learning rate')
    p.add_argument('--patience', type=int, default=10,
                   help='Early stopping patience')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed')
    return p.parse_args()


def load_data(args):
    """Load M4 or synthetic data."""
    if args.use_m4:
        print(f"Loading M4 {args.category} data...")
        loader = M4Loader()
        data = loader.load_category(
            args.category, 
            min_len=200, 
            max_n=args.n_series
        )
        if len(data) == 0:
            print("No M4 data found. Falling back to synthetic.")
            data = create_synthetic_m4(n_series=args.n_series, seed=args.seed)
    else:
        print(f"Creating {args.n_series} synthetic series...")
        data = create_synthetic_m4(n_series=args.n_series, seed=args.seed)
    
    print(f"Loaded {len(data)} series")
    return data


def run_pipeline(data, n_comp=10):
    """
    Run feature engineering pipeline on all series.
    Returns combined train/val/test arrays.
    """
    print("Running feature pipeline...")
    pipe = Pipeline(win_sz=30, n_comp=n_comp)
    
    results = pipe.process_batch(data)
    print(f"Successfully processed {len(results)} series")
    
    if len(results) == 0:
        raise ValueError("No series processed successfully")
    
    combined = pipe.get_combined(results)
    
    print(f"Combined shapes:")
    print(f"  X_train: {combined['X_train'].shape}")
    print(f"  X_val: {combined['X_val'].shape}")
    print(f"  X_test: {combined['X_test'].shape}")
    
    return combined, results


def train_nnknn(combined, args):
    """Train NN-kNN model."""
    import torch
    from src.models.nn_knn_trainer import NNKNNTrainer
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    n_feat = combined['X_train'].shape[1]
    n_cases = combined['X_train'].shape[0]
    
    print(f"\nInitializing NN-kNN: {n_feat} features, {n_cases} cases")
    
    trainer = NNKNNTrainer(
        n_feat=n_feat,
        n_cases=n_cases,
        shared_wts=True,  
        lr=args.lr
    )
    
    print("Training...")
    history = trainer.fit(
        combined['X_train'], combined['y_train'],
        combined['X_val'], combined['y_val'],
        epochs=args.epochs,
        batch_sz=args.batch_sz,
        patience=args.patience,
        verbose=True
    )
    
    return trainer, history


def evaluate_model(trainer, combined):
    """Evaluate trained model on test set."""
    from evaluation.metrics import eval_all
    
    print("\nEvaluating on test set...")
    
    y_pred = trainer.predict(combined['X_test'])
    y_true = combined['y_test']
    y_train = combined['y_train']
    
    y_prev = np.roll(y_true, 1)
    y_prev[0] = y_train[-1] if len(y_train) > 0 else y_true[0]
    
    metrics = eval_all(y_true, y_pred, y_train, y_prev)
    
    print("\n=== Test Results ===")
    print(f"SMAPE: {metrics['smape']:.2f}")
    print(f"MASE: {metrics['mase']:.4f}")
    if metrics['dir_acc'] is not None:
        print(f"Directional Accuracy: {metrics['dir_acc']:.1f}%")
    
    return metrics, y_pred


def run_explanations(trainer, combined, n_samples=5):
    """Generate sample explanations."""
    from src.evaluation.interpretability import NNKNNExplainer
    
    print(f"\n=== Sample Explanations ({n_samples} queries) ===")
    
    explainer = NNKNNExplainer(trainer, k=3)
    
    idx = np.random.choice(len(combined['X_test']), 
                           size=min(n_samples, len(combined['X_test'])), 
                           replace=False)
    X_sample = combined['X_test'][idx]
    
    explanations = explainer.explain_batch(X_sample)
    
    for i, exp in enumerate(explanations):
        print(f"\n--- Query {i+1} ---")
        print(explainer.format_explanation(exp))
    
    consistency = explainer.compute_consistency_score(explanations)
    print(f"\nNeighbor Consistency: mean_std={consistency['mean_neighbor_std']:.4f}")
    
    return explanations


def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    data = load_data(args)
    
    combined, per_series = run_pipeline(data)
    
    trainer, history = train_nnknn(combined, args)
    
    # === DIAGNOSTIC: Check what's happening with activations ===
    import torch
    model = trainer.model
    
    print("\n=== DIAGNOSTIC INFO ===")
    print("Feature weights (min, max):", 
          model.feature_distance.feature_weights.min().item(), 
          model.feature_distance.feature_weights.max().item())
    print("Distance weights (min, max):", 
          model.case_activation.distance_weights.min().item(),
          model.case_activation.distance_weights.max().item())
    
    # Check activation distribution
    X_sample = trainer._to_tensor(combined['X_test'][:10])
    with torch.no_grad():
        delta = model.feature_distance(X_sample, trainer.cases)
        acts = model.case_activation(delta)
        print("Activation range (min, max):", acts.min().item(), acts.max().item())
        print("Activation mean:", acts.mean().item())
        print("Activation std:", acts.std().item())
        
        # Check what sigmoid input looks like
        if model.case_activation.shared_weights:
            ca_input = torch.sum(delta * model.case_activation.distance_weights, dim=2)
        else:
            ca_input = torch.sum(delta * model.case_activation.distance_weights.unsqueeze(0), dim=2)
        ca_input = ca_input + model.case_activation.ca_bias.unsqueeze(0)
        print("Sigmoid INPUT range:", ca_input.min().item(), ca_input.max().item())
    # === END DIAGNOSTIC ===
    
    metrics, y_pred = evaluate_model(trainer, combined)
    
    explanations = run_explanations(trainer, combined)
    
    print("\n=== Experiment Complete ===")
    return {
        'metrics': metrics,
        'history': history,
        'trainer': trainer
    }


if __name__ == "__main__":
    main()