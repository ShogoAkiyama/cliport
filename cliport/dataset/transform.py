import numpy as np


class Transform:

    @classmethod
    def get_random_image_transform_params(cls, image_size, theta_sigma=60):
        theta = np.random.normal(0, np.deg2rad(theta_sigma))

        trans_sigma = np.min(image_size) / 6
        trans = np.random.normal(0, trans_sigma, size=2)  # [x, y]
        pivot = (image_size[1] / 2, image_size[0] / 2)
        return theta, trans, pivot

    @classmethod
    def get_image_transform(cls, theta, trans, pivot=(0, 0)):
        """Compute composite 2D rigid transformation matrix."""
        # Get 2D rigid transformation matrix that rotates an image by theta (in
        # radians) around pivot (in pixels) and translates by trans vector (in
        # pixels)
        pivot_t_image = np.array([
            [1., 0., -pivot[0]],
            [0., 1., -pivot[1]],
            [0., 0., 1.]
        ])
        image_t_pivot = np.array([
            [1., 0., pivot[0]],
            [0., 1., pivot[1]],
            [0., 0., 1.]
        ])
        transform = np.array([
            [np.cos(theta), -np.sin(theta), trans[0]],
            [np.sin(theta), np.cos(theta), trans[1]], [0., 0., 1.]
        ])
        return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))
