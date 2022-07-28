import torch
import kornia


class ImageRotator:
    """Rotate for n rotations."""
    # Reference: https://kornia.readthedocs.io/en/latest/tutorials/warp_affine.html?highlight=rotate

    def __init__(self, n_rotations):
        self.angles = []
        for i in range(n_rotations):
            theta = i * 2 * 180 / n_rotations
            self.angles.append(theta)

    def __call__(self, x_list, pivot, reverse=False):
        rot_x_list = []
        for i, angle in enumerate(self.angles):
            x = x_list[i].unsqueeze(0)

            # create transformation (rotation)
            alpha: float = angle if not reverse else (-1.0 * angle)  # in degrees
            angle: torch.tensor = torch.ones(1) * alpha

            # define the rotation center
            center: torch.tensor = torch.ones(1, 2)
            center[..., 0] = pivot[1]
            center[..., 1] = pivot[0]

            # define the scale factor
            scale: torch.tensor = torch.ones(1, 2)

            # compute the transformation matrix
            M: torch.tensor = kornia.get_rotation_matrix2d(center, angle, scale)

            # apply the transformation to original image
            _, _, h, w = x.shape
            x_warped: torch.tensor = kornia.warp_affine(x.float(), M.to(x.device), dsize=(h, w))
            x_warped = x_warped
            rot_x_list.append(x_warped)

        return rot_x_list
