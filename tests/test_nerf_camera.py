import unittest
import torch
from pytorch3d.renderer import PerspectiveCameras
from camformat3d.nerf_camera import NeRFCamera

class TestNeRFCamera(unittest.TestCase):

    def setUp(self):
        # 这里设置测试用的基本参数
        self.camera_angle_x = torch.tensor([0.69])  # 这个值应当是实际相机参数的合理值
        self.c2w = torch.eye(4)[:3, :4]  # 使用一个简单的单位矩阵作为相机的 world matrix
        self.height = 128
        self.width = 128
        self.in_ndc = True
        self.camera = NeRFCamera(self.camera_angle_x, self.c2w, self.height, self.width, self.in_ndc)

    def test_initialization(self):
        # 测试初始化是否正确
        self.assertEqual(self.camera.camera_angle_x, self.camera_angle_x)
        self.assertTrue(torch.equal(self.camera.c2w, torch.eye(4)[:3, :4]))
        self.assertEqual(self.camera.height, 128)
        self.assertEqual(self.camera.width, 128)
        self.assertTrue(self.camera.in_ndc)

    def test_convert_to_pytorch3d(self):
        # 测试转换到 PyTorch3D 是否成功
        pytorch3d_camera = self.camera.convert_to_pytorch3d()
        self.assertIsInstance(pytorch3d_camera, PerspectiveCameras)

    def test_convert_from_pytorch3d(self):
        # 测试从 PyTorch3D 转换回 NeRFCamera
        pytorch3d_camera = self.camera.convert_to_pytorch3d()
        new_camera = NeRFCamera.convert_from_pytorch3d(pytorch3d_camera, self.height, self.width)
        self.assertIsInstance(new_camera, NeRFCamera)
        self.assertTrue(torch.allclose(new_camera.c2w, self.c2w))

if __name__ == '__main__':
    unittest.main()