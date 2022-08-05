import numpy as np

from cliport.envs.tasks.utils import Utils
from cliport.envs.tasks.transform import Transforms


class PickPlace():
    """Pick and place primitive."""

    def __init__(self, height=0.32, speed=0.01):
        self.height, self.speed = height, speed

    def __call__(self, movep, ee, pose0, pose1):
        """Execute pick and place primitive.
        Args:
            movej: function to move robot joints.
            movep: function to move robot end effector pose.
            ee: robot end effector.
            pose0: SE(3) picking pose.
            pose1: SE(3) placing pose.
    
        Returns:
            timeout: robot movement timed out if True.
        """

        pick_pose, place_pose = pose0, pose1

        # Execute picking primitive.
        prepick_to_pick = ((0, 0, 0.32), (0, 0, 0, 1))
        postpick_to_pick = ((0, 0, self.height), (0, 0, 0, 1))
        prepick_pose = Utils.multiply(pick_pose, prepick_to_pick)
        postpick_pose = Utils.multiply(pick_pose, postpick_to_pick)
        timeout = movep(prepick_pose)

        # Move towards pick pose until contact is detected.
        delta = (
            np.float32([0, 0, -0.001]),
            Transforms.eulerXYZ_to_quatXYZW((0, 0, 0))
        )
        targ_pose = prepick_pose
        while not ee.detect_contact():  # and target_pose[2] > 0:
            targ_pose = Utils.multiply(targ_pose, delta)
            timeout |= movep(targ_pose)
            if timeout:
                return True

        # Activate end effector, move up, and check picking success.
        ee.activate()
        timeout |= movep(postpick_pose, self.speed)
        pick_success = ee.check_grasp()

        # Execute placing primitive if pick is successful.
        if pick_success:
            preplace_to_place = ((0, 0, self.height), (0, 0, 0, 1))
            postplace_to_place = ((0, 0, 0.32), (0, 0, 0, 1))
            preplace_pose = Utils.multiply(place_pose, preplace_to_place)
            postplace_pose = Utils.multiply(place_pose, postplace_to_place)
            targ_pose = preplace_pose
            while not ee.detect_contact():
                targ_pose = Utils.multiply(targ_pose, delta)
                timeout |= movep(targ_pose, self.speed)
                if timeout:
                    return True
            ee.release()
            timeout |= movep(postplace_pose)

        # Move to prepick pose if pick is not successful.
        else:
            ee.release()
            timeout |= movep(prepick_pose)

        return timeout
