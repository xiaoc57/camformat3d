import torch

from einops import rearrange
from pytorch3d.renderer import PerspectiveCameras

class ColmapCamera:
    
    def __init__(self, intrinsics, extrinsics, height, width, in_ndc=True):
        # intrinsics is K, extrinsics is w2c
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.height = height
        self.width = width
        self.T = extrinsics[:3, 3]
        self.R = extrinsics[:3, :3]
        self.in_ndc = in_ndc
        
    def convert_to_pytorch3d(self):
        assert self.in_ndc is True
        
        c2w = torch.inverse(self.extrinsics) # to c2w
        R, T = c2w[:3, :3], c2w[:3, 3:]
        R = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1) # from RDF to LUF for Rotation

        new_c2w = torch.cat([R, T], 1)
        w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0,0,0,1]]).to(new_c2w.device)), 0))
        R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3] # convert R to row-major matrix
        R = R[None] # batch 1 for rendering
        T = T[None] # batch 1 for rendering

        """ Downsample images size for faster rendering """
        H, W = self.height, self.width
        H, W = int(H), int(W)

        intrinsics = self.intrinsics
        # camera['intrinsics'] is 0.5 * H / np.tan(0.5*meta['camera_angle_x'])
        # but we want to compute 1. / np.tan(0.5 * float(meta["camera_angle_x"]))
        # so fcl_ndc = fcl_screen * 2 / min(image_size)

        image_size = ((H, W),)  # (h, w)
        fcl_ndc = ((intrinsics[0] * 2. / min(H, W), intrinsics[1] * 2. / min(H, W)),)
        prp_ndc = (((W / 2 - intrinsics[2]) * 2 / min(H, W), (H / 2 - intrinsics[3]) * 2 / min(H, W)),)
        fcl_screen = ((intrinsics[0], intrinsics[1]),)  # fcl_ndc * min(image_size) / 2
        
        prp_screen = ((intrinsics[2], intrinsics[3]), )  # w / 2 - px_ndc * min(image_size) / 2, h / 2 - py_ndc * min(image_size) / 2
        return PerspectiveCameras(
            focal_length=fcl_ndc, 
            principal_point=prp_ndc, 
            in_ndc=self.in_ndc, 
            R=R, 
            T=T, 
            device=R.device
        )        
    
    @classmethod
    def convert_from_pytorch3d(cls, camera: PerspectiveCameras, height, width):
        Rc = camera.R[0]
        Tc = camera.T[0]

        R = Rc.t()
        T = Tc
        t = torch.Tensor([[0,0,0,1]]).to(camera.device)

        w2c = torch.linalg.inv(torch.cat([torch.cat([R, rearrange(T, 'c -> c 1')], dim = -1), t], dim = 0))
        new_c2w = w2c.clone()

        new_c2w[:, [0, 1]] = w2c[:, [0, 1]] * -1.

        extrinsics = torch.inverse(new_c2w)
        
        H, W = height, width
        H, W = int(H), int(W)
        fcl_ndc = camera.focal_length[0]
        prp_ndc = camera.principal_point[0]
        focal_length_x = fcl_ndc[0] * min(H, W) / 2.
        focal_length_y = fcl_ndc[1] * min(H, W) / 2.
        p1 = W / 2 - prp_ndc[0] * min(H, W) / 2.
        p2 = H / 2 - prp_ndc[1] * min(H, W) / 2.
        intrinsics = [focal_length_x, focal_length_y, p1, p2]
        
        return cls(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            height=height,
            width=width,
            in_ndc=camera.in_ndc
        )
        
        