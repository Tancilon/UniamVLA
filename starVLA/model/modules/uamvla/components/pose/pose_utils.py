import torch
import torch.nn.functional as F


def get_pose_dim(rot_mode: str) -> int:
    """Return pose dimension for a given rotation representation."""
    assert rot_mode in [
        "quat_wxyz",
        "quat_xyzw",
        "euler_xyz",
        "euler_xyz_sx_cx",
        "rot_matrix",
    ], f"the rotation mode {rot_mode} is not supported!"

    if rot_mode in {"quat_wxyz", "quat_xyzw"}:
        return 7
    if rot_mode == "euler_xyz":
        return 6
    if rot_mode in {"euler_xyz_sx_cx", "rot_matrix"}:
        return 9
    raise NotImplementedError


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix via Gram-Schmidt."""
    orig_dtype = d6.dtype
    d6 = d6.float()
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1).to(orig_dtype)


def get_rot_matrix(batch_pose: torch.Tensor, pose_mode: str) -> torch.Tensor:
    """Convert batched rotation representation to rotation matrices."""
    assert pose_mode in [
        "quat_wxyz", "quat_xyzw", "euler_xyz", "euler_xyz_sx_cx", "rot_matrix",
    ], f"the rotation mode {pose_mode} is not supported!"

    if pose_mode in {"quat_wxyz", "quat_xyzw"}:
        if pose_mode == "quat_wxyz":
            quat_wxyz = batch_pose
        else:
            quat_wxyz = batch_pose[:, [3, 0, 1, 2]]
        quat_wxyz = quat_wxyz / torch.norm(quat_wxyz, dim=-1, keepdim=True).clamp_min(1e-12)
        r, i, j, k = torch.unbind(quat_wxyz, dim=-1)
        two_s = 2.0 / (quat_wxyz * quat_wxyz).sum(dim=-1)
        rot_mat = torch.stack(
            (1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
             two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
             two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)),
            dim=-1,
        ).reshape(quat_wxyz.shape[:-1] + (3, 3))
    elif pose_mode == "rot_matrix":
        rot_mat = rotation_6d_to_matrix(batch_pose).transpose(-1, -2)
    elif pose_mode == "euler_xyz_sx_cx":
        rot_sin_theta = batch_pose[:, :3]
        rot_cos_theta = batch_pose[:, 3:6]
        theta = torch.atan2(rot_sin_theta, rot_cos_theta)
        cx, cy, cz = torch.cos(theta[:, 2]), torch.cos(theta[:, 1]), torch.cos(theta[:, 0])
        sx, sy, sz = torch.sin(theta[:, 2]), torch.sin(theta[:, 1]), torch.sin(theta[:, 0])
        rot_mat = torch.stack(
            (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
             sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
             -sy, cy * sx, cy * cx), dim=-1,
        ).reshape(-1, 3, 3)
    elif pose_mode == "euler_xyz":
        cx, cy, cz = torch.cos(batch_pose[:, 2]), torch.cos(batch_pose[:, 1]), torch.cos(batch_pose[:, 0])
        sx, sy, sz = torch.sin(batch_pose[:, 2]), torch.sin(batch_pose[:, 1]), torch.sin(batch_pose[:, 0])
        rot_mat = torch.stack(
            (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
             sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
             -sy, cy * sx, cy * cx), dim=-1,
        ).reshape(-1, 3, 3)
    else:
        raise NotImplementedError
    return rot_mat


def normalize_rotation(rotation: torch.Tensor, rotation_mode: str) -> torch.Tensor:
    """Normalize rotation representation in-place by mode."""
    if rotation_mode in {"quat_wxyz", "quat_xyzw"}:
        rotation /= torch.norm(rotation, dim=-1, keepdim=True).clamp_min(1e-12)
    elif rotation_mode == "rot_matrix":
        rot_matrix = get_rot_matrix(rotation, rotation_mode)
        rotation[:, :3] = rot_matrix[:, :, 0]
        rotation[:, 3:6] = rot_matrix[:, :, 1]
    elif rotation_mode == "euler_xyz_sx_cx":
        rot_sin_theta = rotation[:, :3]
        rot_cos_theta = rotation[:, 3:6]
        theta = torch.atan2(rot_sin_theta, rot_cos_theta)
        rotation[:, :3] = torch.sin(theta)
        rotation[:, 3:6] = torch.cos(theta)
    elif rotation_mode == "euler_xyz":
        pass
    else:
        raise NotImplementedError
    return rotation
