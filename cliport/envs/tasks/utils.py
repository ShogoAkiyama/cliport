import numpy as np
import pybullet as p


class Utils:
    @classmethod
    def multiply(cls, pose0, pose1):
        return p.multiplyTransforms(pose0[0], pose0[1], pose1[0], pose1[1])

    @classmethod
    def invert(cls, pose):
        return p.invertTransform(pose[0], pose[1])

    @classmethod
    def apply(cls, pose, position):
        position = np.float32(position)
        position_shape = position.shape
        position = np.float32(position).reshape(3, -1)
        rotation = np.float32(p.getMatrixFromQuaternion(pose[1])).reshape(3, 3)
        translation = np.float32(pose[0]).reshape(3, 1)
        position = rotation @ position + translation
        return tuple(position.reshape(position_shape))
