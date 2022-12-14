import numpy as np
from transforms3d import euler


class Transforms:

    @classmethod
    def xyz_to_pix(cls, position, bounds, pixel_size):
        u = int(np.round((position[1] - bounds[1, 0]) / pixel_size))
        v = int(np.round((position[0] - bounds[0, 0]) / pixel_size))
        return (u, v)

    @classmethod
    def quatXYZW_to_eulerXYZ(cls, quaternion_xyzw):
        q = quaternion_xyzw
        quaternion_wxyz = np.array([q[3], q[0], q[1], q[2]])
        euler_zxy = euler.quat2euler(quaternion_wxyz, axes='szxy')
        euler_xyz = (euler_zxy[1], euler_zxy[2], euler_zxy[0])
        return euler_xyz

    @classmethod
    def eulerXYZ_to_quatXYZW(cls, rotation):
        euler_zxy = (rotation[2], rotation[0], rotation[1])
        quaternion_wxyz = euler.euler2quat(*euler_zxy, axes='szxy')
        q = quaternion_wxyz
        quaternion_xyzw = (q[1], q[2], q[3], q[0])
        return quaternion_xyzw

    @classmethod
    def pix_to_xyz(cls, pixel, height, bounds, pixel_size, skip_height=False):
        u, v = pixel
        x = bounds[0, 0] + v * pixel_size
        y = bounds[1, 0] + u * pixel_size
        if not skip_height:
            z = bounds[2, 0] + height[u, v]
        else:
            z = 0.0
        return (x, y, z)
