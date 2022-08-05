import os
import random
import string
import tempfile
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

    @classmethod
    def fill_template(cls, assets_root, template, replace):
        """Read a file and replace key strings."""
        full_template_path = os.path.join(assets_root, template)
        with open(full_template_path, 'r') as file:
            fdata = file.read()
        for field in replace:
            for i in range(len(replace[field])):
                fdata = fdata.replace(f'{field}{i}', str(replace[field][i]))
        alphabet = string.ascii_lowercase + string.digits
        rname = ''.join(random.choices(alphabet, k=16))
        tmpdir = tempfile.gettempdir()
        template_filename = os.path.split(template)[-1]
        fname = os.path.join(tmpdir, f'{template_filename}.{rname}')

        with open(fname, 'w') as file:
            file.write(fdata)

        return fname