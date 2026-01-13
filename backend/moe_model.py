# moe_model.py
from typing import List, Dict
import torch
import torch.nn as nn

class SE1D(nn.Module):
    def __init__(self, C: int, reduction: int = 16):
        super().__init__()
        r = max(1, C // reduction)
        self.fc1 = nn.Linear(C, r)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(r, C)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        w = self.fc2(self.act(self.fc1(x)))
        w = self.sig(w)
        return x * w

class ImprovedExpert(nn.Module):
    def __init__(self, in_dim=1536, bottleneck=768, drop=0.3, use_batchnorm=True, use_se=False):
        super().__init__()
        self.use_se = use_se
        
        if use_batchnorm:
            self.pre = nn.Sequential(
                nn.BatchNorm1d(in_dim),
                nn.Linear(in_dim, bottleneck),
                nn.GELU(),
                nn.Dropout(drop),
            )
        else:
            self.pre = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, bottleneck),
                nn.GELU(),
                nn.Dropout(drop),
            )
        
        if use_se:
            self.se = SE1D(bottleneck, reduction=16)
        
        self.mid = nn.Sequential(
            nn.Linear(bottleneck, bottleneck),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.head = nn.Linear(bottleneck, 2)

    def forward(self, x):
        h = self.pre(x)
        if self.use_se:
            h = self.se(h)
        h2 = self.mid(h)
        h = h + h2
        out = self.head(h)
        return out

class TinyGate(nn.Module):
    def __init__(self, in_dim_concat, hidden=64, drop=0.15, n_experts=2, simple=False):
        super().__init__()
        if simple:
            self.net = nn.Sequential(
                nn.Linear(in_dim_concat, n_experts)
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim_concat),
                nn.Linear(in_dim_concat, hidden),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(hidden, n_experts)
            )
    
    def forward(self, x_concat):
        return torch.softmax(self.net(x_concat), dim=1)

class MoEModel(nn.Module):
    def __init__(self, ptms: List[str], in_dim_each=1536, expert_bottleneck=768, expert_drop=0.3,
                 gate_hidden=64, gate_drop=0.15, use_batchnorm=True, use_se=False, 
                 simple_gate=False, stochastic_depth=0.6, use_fusion=False, fusion_dropout=0.5):
        super().__init__()
        self.ptms = ptms
        self.stochastic_depth = stochastic_depth
        self.use_fusion = use_fusion
        
        self.experts = nn.ModuleDict({
            ptm: ImprovedExpert(in_dim_each, expert_bottleneck, expert_drop, use_batchnorm, use_se) 
            for ptm in ptms
        })
        self.gate = TinyGate(in_dim_each * len(ptms), gate_hidden, gate_drop, len(ptms), simple_gate)
        
        if use_fusion:
            self.fusion = nn.Sequential(
                nn.Linear(2, 2),
                nn.Dropout(fusion_dropout)
            )

    def forward(self, x_dict: Dict[str, torch.Tensor]):
        ptm_order = self.ptms
        xs = [x_dict[ptm] for ptm in ptm_order]
        x_concat = torch.cat(xs, dim=1)
        gate_w = self.gate(x_concat)
        
        if self.training and self.stochastic_depth < 1.0:
            keep_prob = torch.full((gate_w.shape[0], len(self.ptms)), self.stochastic_depth, 
                                   device=gate_w.device)
            mask = torch.bernoulli(keep_prob)
            gate_w = gate_w * mask
            gate_w = gate_w / (gate_w.sum(dim=1, keepdim=True) + 1e-8)
        
        expert_logits = torch.stack([self.experts[p](x) for p, x in zip(ptm_order, xs)], dim=1)
        final_logits = (gate_w.unsqueeze(-1) * expert_logits).sum(dim=1)
        
        if self.use_fusion:
            final_logits = self.fusion(final_logits)
        
        return final_logits, expert_logits, gate_w
