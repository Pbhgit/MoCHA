import torch
import torch.nn as nn
import re
from typing import List, Optional
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}
    
class MLPMoE(nn.Module):
    def __init__(self, num_experts, num_selected, mm_channels, channels, num_layers, dropout=False):
        super().__init__()
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.mm_channels = mm_channels # 1024
        # print(self.mm_channels)
        self.channels = channels

        self.gate = nn.Linear(mm_channels, num_experts, bias=False) # [batch_size, num_tokens, channels] -> [batch_size, num_tokens, num_experts]
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(mm_channels, channels), nn.GELU(), nn.Linear(channels, channels)) for _ in range(num_experts)])

    def forward(self, x_img):
      
        gate_logits = self.gate(x_img)
        router_z_loss = torch.logsumexp(gate_logits, dim=-1)
        router_z_loss = torch.square(router_z_loss)
        router_z_loss = router_z_loss.mean()
        gate_softmax = F.softmax(gate_logits, dim=-1, dtype=torch.float).to(x_img.dtype) # [batch_size, num_tokens, num_experts]

        density_1_proxy = reduce(gate_softmax, '... n e -> ... e', 'mean') # [batch_size, num_experts]
        weights, selected_experts = torch.topk(gate_softmax, self.num_selected) # weights, selected_experts: [batch_size, num_tokens, num_selected]
        one_hot_gate_indices = F.one_hot(rearrange(selected_experts, '... k -> k ...'), self.num_experts).float()[0] # [num_selected, batch_size, num_tokens, num_experts] -> [batch_size, num_tokens, num_experts] 
        density_1 = reduce(one_hot_gate_indices, '... n e -> ... e', 'mean') # [batch_size, num_experts]
        balance_loss = (density_1_proxy * density_1).mean() * float(self.num_experts ** 2) 
        weights = weights / torch.sum(weights, dim=-1, keepdim=True).to(x_img.dtype)
        
        results = torch.zeros((x_img.shape[0], x_img.shape[1], self.channels)).to(x_img.device, x_img.dtype)

        for b in range(x_img.shape[0]):
            for i, expert in enumerate(self.experts):
                token_idx, nth_expert = torch.where(selected_experts[b] == i) # 
                results[b][token_idx] += weights[b][token_idx, nth_expert, None] * expert(x_img[b][token_idx])
        
        return results, balance_loss, router_z_loss
    
    @property
    def config(self):
        return {"mm_projector_type": 'smoe_mlp'}

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear') # smoe_mlp
    # print(config.hidden_size)

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size) # hidden_size = 2560

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size * len(config.scales), config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()
    

    elif projector_type == 'smoe_mlp':
        return MLPMoE(num_experts=config.num_experts, num_selected=config.num_selected, mm_channels=(config.mm_hidden_size * len(config.scales)), channels=config.hidden_size, num_layers=config.num_layers, dropout=config.dropout)


    raise ValueError(f'Unknown projector type: {projector_type}')