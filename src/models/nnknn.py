import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDistanceLayer(nn.Module):
    def __init__(self, num_features, num_cases, shared_weights=False):
        super().__init__()
        self.shared_weights = shared_weights
        if shared_weights:
            self.feature_weights = nn.Parameter(torch.ones(num_features))
        else:
            self.feature_weights = nn.Parameter(torch.ones(num_cases, num_features))
    
    def forward(self, q, X):
        B, D = q.shape
        N = X.shape[0]

        q_exp = q.unsqueeze(1)
        X_exp = X.unsqueeze(0)

        delta = (q_exp - X_exp) ** 2       

        w = F.softplus(self.feature_weights)
        if self.shared_weights: 
            delta = delta * w.unsqueeze(0).unsqueeze(0)
        else:
            delta = delta * w.unsqueeze(0)

        return delta


class CaseActivationLayer(nn.Module):
    def __init__(self, num_features, num_cases, shared_weights=False, temp=1.0):
        super().__init__()
        self.shared_weights = shared_weights
        self.temp = temp

        if shared_weights:
            self.distance_weights = nn.Parameter(torch.ones(num_features) * 0.1)
            self.ca_bias = nn.Parameter(torch.ones(num_cases))
        else:
            self.distance_weights = nn.Parameter(torch.ones(num_cases, num_features) * 0.1)
            self.ca_bias = nn.Parameter(torch.ones(num_cases))

    def forward(self, delta):
        B, N, D = delta.shape

        # Force weights negative via -softplus (paper says distance weights <= 0)
        neg_weights = -F.softplus(self.distance_weights)

        if self.shared_weights:
            case_dist = torch.sum(delta * neg_weights.unsqueeze(0).unsqueeze(0), dim=2)
        else:
            case_dist = torch.sum(delta * neg_weights.unsqueeze(0), dim=2)

        # Add positive bias (baseline activation)
        case_activations = case_dist + F.softplus(self.ca_bias).unsqueeze(0)
        
        # Temperature scaling to control sharpness
        case_activations = case_activations / self.temp

        return torch.sigmoid(case_activations)


class TargetAdaptationLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, case_activations, targets):
        ca_normalized = case_activations / (case_activations.sum(dim=1, keepdim=True) + 1e-8)
        y_hat = ca_normalized @ targets
        return y_hat


class NNKNN(nn.Module):
    def __init__(self, num_features, num_cases, shared_weights=False, temp=1.0):
        super().__init__()
        self.feature_distance = FeatureDistanceLayer(num_features, num_cases, shared_weights)
        self.case_activation = CaseActivationLayer(num_features, num_cases, shared_weights, temp)
        self.target_adaptation = TargetAdaptationLayer()

    def forward(self, queries, cases, targets):
        delta = self.feature_distance(queries, cases)
        activations = self.case_activation(delta)
        y_hat = self.target_adaptation(activations, targets)
        return y_hat, activations, delta