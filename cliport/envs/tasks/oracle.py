import numpy as np
import pybullet as p

from cliport.envs.tasks.heightmap import HeightMap
from cliport.envs.utils import render_camera


# Near-orthographic projection.
image_size = (480, 640)
intrinsics = (63e4, 0, 320., 0, 63e4, 240., 0, 0, 1)
position = (0.5, 0, 1000.)
rotation = p.getQuaternionFromEuler((0, np.pi, -np.pi / 2))

oracle_cams = [{
    'image_size': image_size,
    'intrinsics': intrinsics,
    'position': position,
    'rotation': rotation,
    'zrange': (999.7, 1001.),
    'noise': False
}]
bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.3]])
pix_size = 0.003125


def get_true_image():
    """Get RGB-D orthographic heightmaps and segmentation masks."""

    # Capture near-orthographic RGB-D images and segmentation masks.
    color, depth, segm = render_camera(oracle_cams[0])

    # Combine color with masks for faster processing.
    color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

    # Reconstruct real orthographic projection from point clouds.
    hmaps, cmaps = HeightMap.reconstruct_heightmaps(
        [color],
        [depth],
        oracle_cams,
        bounds,
        pix_size
    )

    # Split color back into color and masks.
    cmap = np.uint8(cmaps)[0, Ellipsis, :3]
    hmap = np.float32(hmaps)[0, Ellipsis]
    mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()
    return cmap, hmap, mask
