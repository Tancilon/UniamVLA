"""Point cloud encoders for 6D pose estimation."""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_CLSMSG_CFG = {
    "NPOINTS": [512, 256, 128, None],
    "RADIUS": [[0.02, 0.04], [0.04, 0.08], [0.08, 0.16], [None, None]],
    "NSAMPLE": [[16, 32], [16, 32], [16, 32], [None, None]],
    "MLPS": [
        [[16, 16, 32], [32, 32, 64]],
        [[64, 64, 128], [64, 96, 128]],
        [[128, 196, 256], [128, 196, 256]],
        [[256, 256, 512], [256, 384, 512]],
    ],
}


class PointNet2Simple(nn.Module):
    def __init__(self, input_channels: int = 0, output_dim: int = 1024) -> None:
        super().__init__()
        in_dim = 3 + input_channels
        self.mlp1 = nn.Sequential(nn.Linear(in_dim, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.mlp2 = nn.Sequential(nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        self.mlp3 = nn.Sequential(nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True))
        self.mlp4 = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True))
        self.fc = nn.Sequential(nn.Linear(512, output_dim), nn.BatchNorm1d(output_dim), nn.ReLU(inplace=True))

    def forward(self, pointcloud: torch.Tensor) -> torch.Tensor:
        B, N, _ = pointcloud.shape
        x = pointcloud.reshape(B * N, -1)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = x.reshape(B, N, -1)
        x = x.max(dim=1).values
        x = self.fc(x)
        return x


class Pointnet2ClsMSG(nn.Module):
    def __init__(self, input_channels: int = 0) -> None:
        super().__init__()
        from .pointnet2.pointnet2_modules import PointnetSAModuleMSG
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels
        cfg = _CLSMSG_CFG
        for k in range(len(cfg["NPOINTS"])):
            mlps = [m.copy() for m in cfg["MLPS"][k]]
            channel_out = 0
            for idx in range(len(mlps)):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]
            self.SA_modules.append(
                PointnetSAModuleMSG(npoint=cfg["NPOINTS"][k], radii=cfg["RADIUS"][k],
                                    nsamples=cfg["NSAMPLE"][k], mlps=mlps, use_xyz=True, bn=True))
            channel_in = channel_out

    def forward(self, pointcloud: torch.Tensor) -> torch.Tensor:
        xyz = pointcloud[..., :3].contiguous()
        features = pointcloud[..., 3:].transpose(1, 2).contiguous() if pointcloud.size(-1) > 3 else None
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        return l_features[-1].squeeze(-1)


class PointNet2Wrapper(nn.Module):
    def __init__(self, input_channels: int = 0, output_dim: int = 1024) -> None:
        super().__init__()
        self.use_cuda_backend = False
        if torch.cuda.is_available():
            try:
                self.encoder = Pointnet2ClsMSG(input_channels=input_channels)
                self.use_cuda_backend = True
                logger.info("Using CUDA PointNet2 (Pointnet2ClsMSG) backend.")
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning("CUDA PointNet2 extensions not found (%s). Falling back to PointNet2Simple.", e)
                self.encoder = PointNet2Simple(input_channels=input_channels, output_dim=output_dim)
        else:
            logger.info("CUDA not available. Using PointNet2Simple fallback.")
            self.encoder = PointNet2Simple(input_channels=input_channels, output_dim=output_dim)

    @staticmethod
    def _sync_buffers_to_device(module: nn.Module, device: torch.device) -> None:
        """Move module buffers to the input device without touching parameters.

        DeepSpeed ZeRO-3 owns parameter placement, but non-parameter buffers
        such as BatchNorm running statistics can remain on CPU for small aux
        modules.  PointNet2 receives CUDA point clouds, so stale CPU buffers
        trigger a device mismatch inside ``torch.batch_norm``.
        """
        for child in module.modules():
            for name, buffer in child._buffers.items():
                if buffer is not None and buffer.device != device:
                    child._buffers[name] = buffer.to(device)

    def forward(self, pointcloud: torch.Tensor) -> torch.Tensor:
        self._sync_buffers_to_device(self.encoder, pointcloud.device)
        if self.use_cuda_backend:
            input_dtype = pointcloud.dtype
            return self.encoder(pointcloud.float()).to(input_dtype)
        return self.encoder(pointcloud)
