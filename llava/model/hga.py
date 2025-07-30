import torch
import torch.nn.functional as F

from torch_scatter import scatter_add

def build_innovative_graph_v3(features, group_idx, k=20, cross_weight=0.7):
    device = features.device
    N = features.size(0)

    # adapting temperature 
    temp = 1 / torch.norm(features, dim=1).mean()  
    normed = F.normalize(features, p=2, dim=1)
    sim_matrix = torch.matmul(normed, normed.t()) * torch.exp(temp)

    mask_self = torch.eye(N, dtype=torch.bool, device=device)
    sim_matrix.masked_fill_(mask_self, float('-inf'))

    intra_k = int(k * cross_weight)
    cross_k = k - intra_k

    edge_list = []

    unique_groups = group_idx.unique()
    for g in unique_groups:
        group_mask = (group_idx == g)
        group_size = group_mask.sum().item()

        group_indices = torch.nonzero(group_mask, as_tuple=True)[0]
        row_current = group_indices.unsqueeze(1).expand(-1, k)

        valid_intra_k = min(intra_k, group_size - 1 if group_size > 1 else 0)
        valid_cross_k = min(cross_k, N - group_size)

        # intra-
        intra_mask = group_mask.unsqueeze(1) & group_mask.unsqueeze(0)
        intra_sim = sim_matrix.masked_fill(~intra_mask, float('-inf'))
        intra_sim_group = intra_sim[group_indices]
        if valid_intra_k > 0:
            intra_values, intra_idx = intra_sim_group.topk(valid_intra_k, dim=1)
        else:
            intra_values = torch.empty(group_indices.shape[0], 0, device=device)
            intra_idx = torch.empty(group_indices.shape[0], 0, dtype=torch.long, device=device)

        # inter-
        cross_group_mask = group_mask.unsqueeze(1) & (~group_mask.unsqueeze(0))
        cross_sim = sim_matrix.masked_fill(~cross_group_mask, float('-inf'))
        cross_sim_group = cross_sim[group_indices]
        if valid_cross_k > 0:
            cross_values, cross_idx = cross_sim_group.topk(valid_cross_k, dim=1)
        else:
            cross_values = torch.empty(group_indices.shape[0], 0, device=device)
            cross_idx = torch.empty(group_indices.shape[0], 0, dtype=torch.long, device=device)

        combined_idx = torch.cat([intra_idx, cross_idx], dim=1)
        combined_values = torch.cat([intra_values, cross_values], dim=1)

        row = row_current[:, :combined_idx.size(1)] if combined_idx.size(1) > 0 else torch.empty(0, k, device=device)
        row = row.reshape(-1) if row.numel() > 0 else torch.empty(0, device=device)
        col = combined_idx.reshape(-1) if combined_idx.numel() > 0 else torch.empty(0, device=device)
        val = combined_values.reshape(-1) if combined_values.numel() > 0 else torch.empty(0, device=device)
        edge_list.append((row, col, val))

    row_index = torch.cat([e[0] for e in edge_list]) if edge_list else torch.empty(0, device=device)
    col_index = torch.cat([e[1] for e in edge_list]) if edge_list else torch.empty(0, device=device)
    edge_sim = torch.cat([e[2] for e in edge_list]) if edge_list else torch.empty(0, device=device)

    if row_index.numel() == 0:
        return torch.empty((2, 0), device=device), torch.empty(0, device=device)

    eps = 1e-8
    edge_min, edge_max = edge_sim.min(), edge_sim.max()
    base_weight = (edge_sim - edge_min) / (edge_max - edge_min + eps)
    edge_weight = base_weight

    edge_index1 = torch.stack([row_index, col_index], dim=0)
    edge_index2 = torch.stack([col_index, row_index], dim=0)
    edge_index = torch.cat([edge_index1, edge_index2], dim=1)

    edge_weight1 = edge_weight
    edge_weight2 = edge_weight.clone()
    edge_weight = torch.cat([edge_weight1, edge_weight2], dim=0)

    return edge_index, edge_weight


def innovative_graph_propagation_v3(x, edge_index, edge_weight):
    N = x.size(0)
    row, col = edge_index

    w_sum = scatter_add(edge_weight, row, dim=0, dim_size=N) + 1e-8
    w_norm = edge_weight / w_sum[row]
    x_col = x[col] * w_norm.unsqueeze(1)
    x_agg = scatter_add(x_col, row, dim=0, dim_size=N)


    feature_diff = torch.abs(x - x_agg).mean(dim=1, keepdim=True)
    gate = torch.sigmoid(10 * (feature_diff - 0.2))

    x_new = (1 - gate) * x + gate * x_agg

    x = F.layer_norm(x_new, x_new.shape[1:])

    return x