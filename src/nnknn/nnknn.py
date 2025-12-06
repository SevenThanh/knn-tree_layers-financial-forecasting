import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistanceLayer(nn.Module):
    """
    Compute weighted feature-wise distances between:
       - query vector q:      [B, D]
       - stored cases X:      [N, D]

    Produces:
       - distances: [B, N]
    """
    def __init__(self, num_features):
        super().__init__()
        # Feature weights (positive)
        self.feature_weights = nn.Parameter(torch.ones(num_features))

    def forward(self, q, X):
        # q: [B, D], X: [N, D]
        B, D = q.shape
        N = X.shape[0]

        w = F.softplus(self.feature_weights)  # ensure positivity

        # Expand for broadcasting
        q_exp = q.unsqueeze(1).expand(B, N, D)     # [B, N, D]
        X_exp = X.unsqueeze(0).expand(B, N, D)     # [B, N, D]

        # Weighted L1 distance (paper uses L1 or L2)
        dist = torch.sum(w * torch.abs(q_exp - X_exp), dim=2)   # [B, N]
        return dist


class CaseActivationLayer(nn.Module):
    """
    Converts distances → activations using a learned beta.
    """
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))  # inverse temperature

    def forward(self, distances):
        # soft activation: exp(-beta * distance)
        beta = F.softplus(self.beta)
        activations = torch.exp(-beta * distances)
        return activations  # [B, N]


class NNkNNRegression(nn.Module):
    def __init__(self, stored_cases, stored_targets, k=10, out_dim=1):
        """
        stored_cases: [N, D] tensor of prototype cases
        stored_targets: [N, out_dim] regression labels for each stored case
        """
        super().__init__()

        self.X = nn.Parameter(stored_cases, requires_grad=False)
        self.y = nn.Parameter(stored_targets, requires_grad=False)

        self.k = k
        self.D = stored_cases.shape[1]
        self.out_dim = out_dim

        # Layers
        self.feature_distance = FeatureDistanceLayer(self.D)
        self.case_activation = CaseActivationLayer()

        # Optional small MLP for final regression
        self.output_layer = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )

    def forward(self, q):
        """
        q: [B, D]
        returns predictions: [B, out_dim]
        """

        # Step 1 — Compute weighted feature distance
        dist = self.feature_distance(q, self.X)       # [B, N]

        # Step 2 — Convert to case activation
        act = self.case_activation(dist)              # [B, N]

        # Step 3 — Soft k-NN: pick top-k activations (soft, differentiable)
        topk_vals, topk_idx = torch.topk(act, self.k, dim=1)  # [B, k]

        # Normalize weights to sum to 1
        knn_weights = F.softmax(topk_vals, dim=1)            # [B, k]

        # Gather k target neighbors
        yk = self.y[topk_idx]                                # [B, k, out_dim]

        # Weighted sum → regression prediction
        pred = torch.sum(knn_weights.unsqueeze(2) * yk, dim=1)  # [B, out_dim]

        # Optional refinement MLP
        pred = self.output_layer(pred)

        return pred
