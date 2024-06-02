import torch
import numpy as np

from einops import rearrange
from pytorch3d.renderer import PerspectiveCameras

class NeRFCamera:

    def __init__(self, camera_angle_x, c2w, height, width, in_ndc=True):
        self.camera_angle_x = camera_angle_x
        self.c2w = c2w  # [4, 4]
        self.in_ndc = in_ndc
        self.height = height
        self.width = width

        if in_ndc:
            self.focal_length = torch.FloatTensor([
                1. / np.tan(0.5 * float(self.camera_angle_x)),
                1. / np.tan(0.5 * float(self.camera_angle_x)),
            ])
            self.principal_point = torch.FloatTensor([0.0, 0.0])
        else:
            raise NotImplementedError("nerf camera in_ndc must != False")


    def convert_to_pytorch3d(self):
        assert self.in_ndc is True

        pose_target = self.c2w[:3, :4]
        mtx = torch.eye(4, dtype=pose_target.dtype)
        mtx[:3, :3] = pose_target[:3, :3].t()
        mtx[3, :3] = pose_target[:, 3]
        mtx = mtx.inverse()
        # flip the XZ coordinates.
        mtx[:, [0, 2]] = mtx[:, [0, 2]] * -1.0

        Rpt3, Tpt3 = mtx[:, :3].split([3, 1], dim=0)

        focal_length_pt3 = rearrange(self.focal_length, "c -> 1 c")
        principal_point_pt3 = rearrange(self.principal_point, "c -> 1 c")

        return PerspectiveCameras(
            focal_length=focal_length_pt3,
            principal_point=principal_point_pt3,
            R=Rpt3[None],
            T=Tpt3,
            in_ndc=self.in_ndc
        )
    
    @classmethod
    def convert_from_pytorch3d(cls, camera: PerspectiveCameras, height, width):
        Rpt3, Tpt3 = camera.R, camera.T
        b_ = Rpt3.shape[0]
        tmp = torch.cat([Rpt3, rearrange(Tpt3, "n c -> n 1 c")], dim=1)
        b = torch.cat([torch.zeros(b_, 3, 1), torch.ones(b_, 1, 1)], dim=1)
        tmp = torch.cat([tmp, b], dim=-1)
        tmp[..., [0,2]] = tmp[..., [0,2]] * -1.0

        tmp = tmp.inverse()

        c2w = torch.cat([torch.transpose(tmp[..., :3, :3], 1, 2), rearrange(tmp[...,3, :3], "n c -> n c 1")], dim=2)

        # Calculate camera_angle_x from the focal length
        # Placeholder calculation: this needs to be correctly calculated from camera's focal length
        camera_angle_x = np.arctan(1.0 / float(camera.focal_length[0, 0])) * 2

        return cls(camera_angle_x, c2w, height, width, in_ndc=camera.in_ndc)


