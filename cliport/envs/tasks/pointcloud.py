import numpy as np


class PointCloud:

    @classmethod
    def get_pointcloud(cls, depth, intrinsics):
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
