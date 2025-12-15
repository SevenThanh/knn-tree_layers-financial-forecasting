import torch
import torch.nn as nn
import torch.nn.functional as F
    
'''
Original NNkNN for regression:
- no topk selection layer as recommended by paper for regression
- weight sharing off
'''

class FeatureDistanceLayer(nn.Module):
    def __init__(self, num_features, num_cases, shared_weights=False):
        super().__init__()
        self.shared_weights = shared_weights
        #Feature weights for shared case: 1 weight per feature
        if shared_weights:
            self.feature_weights = nn.Parameter(torch.ones(num_features))
        #Feature weights for unstared case: weights (with dimentions: num_features) for each case
        else:
            self.feature_weights = nn.Parameter(torch.ones(num_cases, num_features))
    
    def forward(self, q, X):
        # dimensions
        B, D = q.shape                                  # B: batch size, D: number of features
        N = X.shape[0]                                  # N: number of cases

        # prepare tensors for broadcasting
        q_exp = q.unsqueeze(1)                          # [B, 1, D]
        X_exp = X.unsqueeze(0)                          # [1, N, D] 

        # L2 distance
        delta = (q_exp - X_exp) ** 2       

        # linear combination of distances and weights
        w = F.softplus(self.feature_weights)            # [D] for shared case, [N, D] for unshared case
        if self.shared_weights: 
            delta = delta * w.unsqueeze(0).unsqueeze(0) # [B, N, D]
        else:
            delta = delta * w.unsqueeze(0)              # [B, N, D]

        return delta

class CaseActivationLayer(nn.Module):
    def __init__(self, num_features, num_cases, shared_weights=False):
        super().__init__()
        self.shared_weights = shared_weights
        
        # for large case bases, need strong pos bias to offset neg distances
        if num_cases > 2000:
            init_bias = 0.0  # strong pos bias for 3k+ cases
        elif num_cases > 1000:
            init_bias = 0.0
        else:
            init_bias = 0.0
        
        if shared_weights:
            self.distance_weights = nn.Parameter(torch.randn(num_features) * 0.05 - 0.15)
            self.ca_bias = nn.Parameter(torch.ones(num_cases) * init_bias)
        else:
            self.distance_weights = nn.Parameter(torch.randn(num_cases, num_features) * 0.05 - 0.15)
            self.ca_bias = nn.Parameter(torch.ones(num_cases) * init_bias)




    def forward(self, delta):
        # delta dimensions
        B, N, D = delta.shape

        # FORCE NEGATIVE WEIGHTS: Ensure distance acts as a penalty
        # We use -abs(w) so that larger distance always reduces activation
        w = -torch.abs(self.distance_weights)

        # linear combination of distances and distance weights
        if self.shared_weights:
            case_activations = torch.sum(
                delta * w.unsqueeze(0).unsqueeze(0), 
                dim=2
            )                                                                           # [B, N]
        else:
            case_activations = torch.sum(
                delta * w.unsqueeze(0), 
                dim=2
            )                                                                           # [B, N]

        case_activations = case_activations + self.ca_bias.unsqueeze(0)                 # [B, N]
        return torch.sigmoid(case_activations)                                          # [B, N]                                         # [B, N]

class TargetAdaptationLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, case_activations, targets):
        """
        case_activations: [B, N]
        targets:          [N, D] (same as X)
        Returns:
            y_hat: [B, D] predicted values per query
        """

        # normalize activations per query
        ca_normalized = case_activations / (case_activations.sum(dim=1, keepdim=True) + 1e-2)   # [B, N]

        # weighted sum over cases
        y_hat = ca_normalized @ targets                                                         # [B, N] @ [N, D] → [B, D]

        return y_hat


class NNKNN(nn.Module):
    def __init__(self, num_features, num_cases, shared_weights=False):
        super().__init__()
        self.feature_distance = FeatureDistanceLayer(num_features, num_cases, shared_weights)
        self.case_activation = CaseActivationLayer(num_features, num_cases, shared_weights)
        self.target_adaptation = TargetAdaptationLayer()

    def forward(self, queries, cases, targets, q_indices=None):
        """
        queries: [B, D] query batch
        cases:   [N, D] stored cases
        targets: [N, C] regression targets for cases
        Returns:
            y_hat: [B, C] predictions
            activations: [B, N]
            distances: [B, N, D]
        """
        # 1. Feature distances
        delta = self.feature_distance(queries, cases)  # [B, N, D]
        # 2. Case activations
        activations = self.case_activation(delta)     # [B, N]
        if q_indices is not None:
            mask = torch.ones_like(activations) 
            mask.scatter_(1, q_indices.view(-1, 1), 0.0)
            activations = activations * mask
        # 3. Weighted sum for regression
        y_hat = self.target_adaptation(activations, targets)  # [B, C]

        return y_hat, activations, delta