import numpy as np

class RollingNormalizer:
    def __init__(self, win_sz=30, eps=1e-8, def_std=0.01, use_expand=False):
        self.win_sz = win_sz
        self.eps = eps
        self.def_std = def_std
        self.use_expand = use_expand
        self.max_ts_used = []
        self.stats = []
