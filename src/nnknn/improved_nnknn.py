import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, activation=nn.ReLU):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NNkNNRegression(nn.Module):
    """
    Glocal NN-kNN for regression (local + global + gate).

    Key features:
      - local embedding encoder -> soft kNN lookup over memory
      - global encoder -> global regression estimate
      - gate g(x) in [0,1] blending local & global predictions
      - neighbor dropout (training only)
      - top-k selection (uses topk on activations; gradients flow through activations)
      - optional local linear correction head
    """

    def __init__(
        self,
        stored_cases: torch.Tensor,
        stored_targets: torch.Tensor,
        input_dim: int,
        embed_dim: int = 64,
        global_hidden: Optional[list] = None,
        local_hidden: Optional[list] = None,
        k: int = 16,
        neighbor_dropout: float = 0.1,
        use_local_linear: bool = False,
        local_linear_hidden: Optional[list] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
          stored_cases: (N, D) tensor of memory inputs (float)
          stored_targets: (N, T) tensor of memory targets (float)
          input_dim: original input dim D
          embed_dim: embedding dim E
          global_hidden: hidden dims for global MLP (if None => [128,64])
          local_hidden: hidden dims for local embedding MLP (if None => [128])
          k: number of neighbors
          neighbor_dropout: probability to drop a neighbor during training
          use_local_linear: whether to apply a small local linear correction
          local_linear_hidden: hidden dims for local linear MLP (if None => [32])
        """
        super().__init__()

        device = device or torch.device("cpu")

        # memory (store as buffers so they move with .to(device))
        self.register_buffer("memory_inputs", stored_cases.to(device))
        self.register_buffer("memory_targets", stored_targets.to(device))

        # dims
        self.N = stored_cases.shape[0]
        self.D = input_dim
        self.T = stored_targets.shape[1]
        self.E = embed_dim
        self.k = k
        self.neighbor_dropout = neighbor_dropout
        self.use_local_linear = use_local_linear

        # Local encoder (shared for queries and memory)
        if local_hidden is None:
            local_hidden = [embed_dim]
        self.local_encoder = MLP(self.D, local_hidden, self.E)

        # Per-dimension (embedding dim) feature weights (diagonal)
        # Keep unconstrained param and apply softplus in forward to ensure positive
        self.feature_weights = nn.Parameter(torch.ones(self.E) * 0.1)

        # Learnable temperature (inverse of beta)
        self.inv_temp = nn.Parameter(torch.tensor(1.0))  # we will use softplus

        # Local aggregation refinement MLP (optional)
        # Takes aggregated neighbor target (or combined) and returns delta
        if self.use_local_linear:
            if local_linear_hidden is None:
                local_linear_hidden = [32]
            # Input to local linear head: [query_emb_dim + aggregated_target_dim]
            self.local_linear_head = MLP(self.E + self.T, local_linear_hidden, self.T)
        else:
            self.local_linear_head = None

        # Optional tiny output refinement for the combined prediction
        self.refinement_head = MLP(self.T, [max(32, self.T * 4)], self.T)

        # Global encoder + global regressor
        if global_hidden is None:
            global_hidden = [128, 64]
        self.global_encoder = MLP(self.D, global_hidden, 64)
        self.global_regressor = MLP(64, [64], self.T)

        # Gate network g(x) -> scalar in [0,1]
        self.gate = nn.Sequential(
            nn.Linear(64 + self.E, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # small epsilon for numerical stability
        self.eps = 1e-9

    def compute_local_embeddings(self, x):
        # x: (B, D) or (N, D)
        return self.local_encoder(x)  # -> (B, E) or (N, E)

    def pairwise_weighted_distance(self, q_emb, mem_emb):
        """
        q_emb: (B, E)
        mem_emb: (N, E)
        returns: distances (B, N)
        Weighted squared differences with per-dim positive weights.
        """
        # ensure positive weights
        w = F.softplus(self.feature_weights)  # (E,)
        # apply sqrt(w) to the embeddings so that sum((sqrt(w)*(q-c))^2) = sum(w*(q-c)^2)
        sqrt_w = torch.sqrt(w + self.eps)
        qw = q_emb * sqrt_w.unsqueeze(0)  # (B, E)
        mw = mem_emb * sqrt_w.unsqueeze(0)  # (N, E)
        # pairwise squared distances via efficient broadcasting or cdist
        # using cdist on transformed vectors:
        # returns (B, N) with squared euclidean distances
        dists = torch.cdist(qw, mw, p=2)  # euclidean
        dists = dists.pow(2)
        return dists  # (B, N)

    def soft_topk_activations(self, activations, topk_idx=None):
        """
        activations: (B, N) raw positive activations (e.g., exp(-d*beta))
        We select top-k values (per row) and renormalize them with softmax over the k selected.
        Returns:
          topk_vals: (B, k) positive values
          topk_idx: (B, k) indices
          topk_weights: (B, k) normalized weights summing to 1
        """
        # If topk_idx provided (e.g. precomputed), use it; otherwise compute.
        B = activations.shape[0]
        if topk_idx is None:
            topk_vals, topk_idx = torch.topk(activations, self.k, dim=1)
        else:
            topk_vals = torch.gather(activations, 1, topk_idx)

        # optionally apply neighbor dropout during training by zeroing some topk entries
        if self.training and self.neighbor_dropout > 0.0:
            mask = torch.bernoulli(torch.ones_like(topk_vals) * (1 - self.neighbor_dropout)).to(topk_vals.device)
            topk_vals = topk_vals * mask

            # If all dropped in a row, add tiny const to avoid all zeros
            zero_rows = (topk_vals.sum(dim=1) == 0.0)
            if zero_rows.any():
                topk_vals[zero_rows, :] = topk_vals[zero_rows, :] + 1e-6

        # normalize with softmax so they sum to 1
        topk_weights = torch.softmax(topk_vals, dim=1)  # (B, k)
        return topk_vals, topk_idx, topk_weights

    def forward(self, x):
        """
        x: (B, D) queries
        returns: predictions (B, T), optionally also returns gate and local/global preds
        """
        B = x.shape[0]
        # device = x.device

        # ---- Local path ----
        # embeddings
        q_emb = self.compute_local_embeddings(x)                           # (B, E)
        mem_emb = self.compute_local_embeddings(self.memory_inputs)        # (N, E)

        # distances and activations
        dists = self.pairwise_weighted_distance(q_emb, mem_emb)            # (B, N)

        # temperature -> use softplus to ensure positive
        temp = F.softplus(self.inv_temp) + self.eps                         # scalar > 0
        # activation = exp(-d / temp)
        raw_acts = torch.exp(-dists / (temp + self.eps))                   # (B, N)

        # select top-k activations -> weights
        topk_vals, topk_idx, topk_weights = self.soft_topk_activations(raw_acts)

        # gather neighbor targets and neighbor embeddings
        # topk_idx: (B, k)
        neigh_targets = self.memory_targets[topk_idx]   # (B, k, T)
        neigh_embs = mem_emb[topk_idx]                 # (B, k, E)

        # Weighted aggregation of neighbor targets (local prediction)
        # If targets are multi-dim, this broadcasts correctly
        local_agg = torch.sum(topk_weights.unsqueeze(2) * neigh_targets, dim=1)  # (B, T)

        # optional local linear correction: use q_emb and weighted mean of neighbor embeddings
        if self.use_local_linear:
            neigh_emb_mean = torch.sum(topk_weights.unsqueeze(2) * neigh_embs, dim=1)  # (B, E)
            local_linear_input = torch.cat([q_emb, local_agg], dim=1)  # (B, E+T) — choose this combination
            local_delta = self.local_linear_head(local_linear_input)  # (B, T)
            local_pred = local_agg + local_delta
        else:
            local_pred = local_agg  # (B, T)

        # ---- Global path ----
        global_repr = self.global_encoder(x)     # (B, 64)
        global_pred = self.global_regressor(global_repr)  # (B, T)

        # ---- Gate ----
        # Build gating input: concat global_repr and aggregated local embedding summary (e.g., q_emb)
        gate_input = torch.cat([global_repr, q_emb], dim=1)  # (B, 64+E)
        g = self.gate(gate_input).squeeze(1)                 # (B,) in (0,1)

        # final blended pred
        g = g.unsqueeze(1)  # (B, 1)
        blended = g * local_pred + (1.0 - g) * global_pred   # (B, T)

        # optional refinement
        out = self.refinement_head(blended)  # (B, T)

        # return prediction and diagnostics
        return out, {
            "gate": g.squeeze(1),                  # (B,)
            "local_pred": local_pred.detach(),
            "global_pred": global_pred.detach(),
            "topk_idx": topk_idx
        }

