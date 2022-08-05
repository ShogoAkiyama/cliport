import numpy as np
import pybullet as p

from cliport.envs.tasks.pointcloud import PointCloud


class HeightMap:

    @classmethod
    def reconstruct_heightmaps(cls, color, depth, configs, bounds, pixel_size):
        """Reconstruct top-down heightmap views from multiple 3D pointclouds."""
        heightmaps, colormaps = [], []
        for color, depth, config in zip(color, depth, configs):
            intrinsics = np.array(config['intrinsics']).reshape(3, 3)
            xyz = PointCloud.get_pointcloud(depth, intrinsics)
            position = np.array(config['position']).reshape(3, 1)
            rotation = p.getMatrixFromQuaternion(config['rotation'])
            rotation = np.array(rotation).reshape(3, 3)
            transform = np.eye(4)
            transform[:3, :] = np.hstack((rotation, position))
            xyz = PointCloud.transform_pointcloud(xyz, transform)
            heightmap, colormap = cls.get_heightmap(xyz, color, bounds, pixel_size)
            heightmaps.append(heightmap)
            colormaps.append(colormap)
        return heightmaps, colormaps

    @classmethod
    def get_heightmap(cls, points, colors, bounds, pixel_size):
        """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.

        Args:
            points: HxWx3 float array of 3D points in world coordinates.
            colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
            bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
                region in 3D space to generate heightmap in world coordinates.
            pixel_size: float defining size of each pixel in meters.

        Returns:
            heightmap: HxW float array of height (from lower z-bound) in meters.
            colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
        """
        width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
        height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
        heightmap = np.zeros((height, width), dtype=np.float32)
        colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

        # Filter out 3D points that are outside of the predefined bounds.
        ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
        iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
        iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
        valid = ix & iy & iz
        points = points[valid]
        colors = colors[valid]

        # Sort 3D points by z-value, which works with array assignment to simulate
        # z-buffering for rendering the heightmap image.
        iz = np.argsort(points[:, -1])
        points, colors = points[iz], colors[iz]
        px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
        py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
        px = np.clip(px, 0, width - 1)
        py = np.clip(py, 0, height - 1)
        heightmap[py, px] = points[:, 2] - bounds[2, 0]
        for c in range(colors.shape[-1]):
            colormap[py, px, c] = colors[:, c]
        return heightmap, colormap

    @classmethod
    def get_fused_heightmap(cls, obs, configs, bounds, pix_size):
        """Reconstruct orthographic heightmaps with segmentation masks."""
        heightmaps, colormaps = cls.reconstruct_heightmaps(
            obs['color'],
            obs['depth'],
            configs,
            bounds,
            pix_size
        )
        colormaps = np.float32(colormaps)
        heightmaps = np.float32(heightmaps)

        # Fuse maps from different views.
        valid = np.sum(colormaps, axis=3) > 0
        repeat = np.sum(valid, axis=0)
        repeat[repeat == 0] = 1
        cmap = np.sum(colormaps, axis=0) / repeat[Ellipsis, None]
        cmap = np.uint8(np.round(cmap))
        hmap = np.max(heightmaps, axis=0)  # Max to handle occlusions.
        return cmap, hmap
