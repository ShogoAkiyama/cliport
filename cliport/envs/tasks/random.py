import numpy as np


class Random:

    @classmethod
    def get_random_size(cls, min_x, max_x, min_y, max_y, min_z, max_z):
        """Get random box size."""
        size = np.random.rand(3)
        size[0] = size[0] * (max_x - min_x) + min_x
        size[1] = size[1] * (max_y - min_y) + min_y
        size[2] = size[2] * (max_z - min_z) + min_z
        return tuple(size)

    @classmethod
    def sample_distribution(cls, prob, n_samples=1):
        """Sample data point from a custom distribution."""
        flat_prob = prob.flatten() / np.sum(prob)
        rand_ind = np.random.choice(
            np.arange(len(flat_prob)),
            n_samples,
            p=flat_prob,
            replace=False
        )
        rand_ind_coords = np.array(np.unravel_index(rand_ind, prob.shape)).T
        return np.int32(rand_ind_coords.squeeze())
