import numpy as np


class PointCloud:

    @classmethod
    def get_pointcloud(cls, depth, intrinsics):
        """Get 3D pointcloud from perspective depth image.

        Args:
            depth: HxW float array of perspective depth in meters.
            intrinsics: 3x3 float array of camera intrinsics matrix.

        Returns:
            points: HxWx3 float array of 3D points in camera coordinates.
        """
        height, width = depth.shape
        xlin = np.linspace(0, width - 1, width)
        ylin = np.linspace(0, height - 1, height)
        px, py = np.meshgrid(xlin, ylin)
        px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
        py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
        points = np.float32([px, py, depth]).transpose(1, 2, 0)
        return points

    @classmethod
    def transform_pointcloud(cls, points, transform):
        """Apply rigid transformation to 3D pointcloud.

        Args:
            points: HxWx3 float array of 3D points in camera coordinates.
            transform: 4x4 float array representing a rigid transformation matrix.

        Returns:
            points: HxWx3 float array of transformed 3D points.
        """
        padding = ((0, 0), (0, 0), (0, 1))
        homogen_points = np.pad(
            points.copy(),
            padding,
            'constant',
            constant_values=1
        )
        for i in range(3):
            points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
        return points
