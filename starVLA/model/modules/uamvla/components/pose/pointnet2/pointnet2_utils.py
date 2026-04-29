"""PointNet2 utility functions — pure PyTorch implementation.

Replaces the CUDA C++ extension (pointnet2_cuda) with native PyTorch ops.
This avoids misaligned-address errors on H200 (sm_90) and works on any GPU
architecture.  For num_points ≤ 4096 the performance difference is negligible.
"""

import torch
import torch.nn as nn
from typing import Tuple


# ---------------------------------------------------------------------------
#  Core ops — pure PyTorch, fully differentiable
# ---------------------------------------------------------------------------

def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Iterative furthest point sampling.

    :param xyz: (B, N, 3)
    :param npoint: number of samples
    :return: (B, npoint) int64 indices
    """
    B, N, _ = xyz.size()
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, dtype=torch.float32, device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_idx = torch.arange(B, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz.float() - centroid.float()) ** 2, dim=-1)  # (B, N)
        distance = torch.min(distance, dist)
        farthest = distance.argmax(dim=-1)

    return centroids


def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather features by index.

    :param features: (B, C, N)
    :param idx: (B, npoint) long or int indices
    :return: (B, C, npoint)
    """
    idx_long = idx.long()
    B, C, N = features.size()
    npoint = idx_long.size(1)
    idx_expanded = idx_long.unsqueeze(1).expand(B, C, npoint)  # (B, C, npoint)
    return torch.gather(features, 2, idx_expanded)


def ball_query(radius: float, nsample: int, xyz: torch.Tensor,
               new_xyz: torch.Tensor) -> torch.Tensor:
    """Find nsample points within radius for each query point.

    :param radius: ball radius
    :param nsample: max number of neighbours
    :param xyz: (B, N, 3) all points
    :param new_xyz: (B, npoint, 3) query centres
    :return: (B, npoint, nsample) int64 indices into xyz
    """
    B, N, _ = xyz.size()
    npoint = new_xyz.size(1)
    device = xyz.device

    # Pairwise squared distances: (B, npoint, N)
    dists = torch.cdist(new_xyz.float(), xyz.float()).pow(2)

    # Mask outside radius
    mask = dists > radius * radius
    dists[mask] = float('inf')

    # Sort and take nsample nearest
    _, sort_idx = dists.sort(dim=-1)              # (B, npoint, N)
    idx = sort_idx[:, :, :nsample]                # (B, npoint, nsample)

    # Where we got fewer than nsample neighbours, repeat the first neighbour
    first = sort_idx[:, :, 0:1].expand_as(idx)
    invalid = dists.gather(-1, idx) == float('inf')
    idx[invalid] = first[invalid]

    return idx


def grouping_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Group features by multi-dim index.

    :param features: (B, C, N)
    :param idx: (B, npoint, nsample) long or int indices
    :return: (B, C, npoint, nsample)
    """
    idx_long = idx.long()
    B, C, N = features.size()
    npoint, nsample = idx_long.size(1), idx_long.size(2)
    idx_flat = idx_long.reshape(B, -1)                           # (B, npoint*nsample)
    idx_expanded = idx_flat.unsqueeze(1).expand(B, C, -1)        # (B, C, npoint*nsample)
    grouped = torch.gather(features, 2, idx_expanded)            # (B, C, npoint*nsample)
    return grouped.reshape(B, C, npoint, nsample)


def three_nn(unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find 3 nearest neighbours of unknown in known.

    :param unknown: (B, N, 3)
    :param known: (B, M, 3)
    :return:
        dist: (B, N, 3) L2 distances
        idx: (B, N, 3) int64 indices
    """
    dists = torch.cdist(unknown.float(), known.float())  # (B, N, M)
    dist2, idx = dists.topk(3, dim=-1, largest=False)     # (B, N, 3)
    return dist2, idx


def three_interpolate(features: torch.Tensor, idx: torch.Tensor,
                      weight: torch.Tensor) -> torch.Tensor:
    """Weighted interpolation using 3 nearest neighbours.

    :param features: (B, C, M)
    :param idx: (B, N, 3) int64 indices
    :param weight: (B, N, 3) interpolation weights
    :return: (B, C, N)
    """
    B, C, M = features.size()
    N = idx.size(1)

    # Gather the 3 neighbour features: (B, C, N, 3)
    idx_expanded = idx.long().unsqueeze(1).expand(B, C, N, 3)
    feat_expanded = features.unsqueeze(3).expand(B, C, M, 3)
    # Simpler: index directly
    gathered = torch.zeros(B, C, N, 3, device=features.device, dtype=features.dtype)
    for j in range(3):
        idx_j = idx[:, :, j].long().unsqueeze(1).expand(B, C, N)  # (B, C, N)
        gathered[:, :, :, j] = torch.gather(features, 2, idx_j)

    # Weighted sum: (B, C, N)
    w = weight.unsqueeze(1)  # (B, 1, N, 3)
    return (gathered * w).sum(dim=-1)


# ---------------------------------------------------------------------------
#  Grouping modules (same API as CUDA version)
# ---------------------------------------------------------------------------

class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor,
                features: torch.Tensor = None) -> torch.Tensor:
        """
        :param xyz: (B, N, 3)
        :param new_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N)
        :return: (B, 3+C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)

        xyz_trans = xyz.transpose(1, 2).contiguous()          # (B, 3, N)
        grouped_xyz = grouping_operation(xyz_trans, idx)       # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz: torch.Tensor, new_xyz: torch.Tensor,
                features: torch.Tensor = None) -> torch.Tensor:
        """
        :param xyz: (B, N, 3)
        :param new_xyz: ignored
        :param features: (B, C, N)
        :return: (B, C+3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
