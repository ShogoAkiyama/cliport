import os
import collections
import numpy as np
import cv2
import random
import string
import tempfile
import pybullet as p

from cliport.envs.tasks.utils import Utils
from cliport.envs.tasks.transform import Transforms
from cliport.envs.tasks.primitives import PickPlace
from cliport.envs.tasks.grippers import Suction
from cliport.envs.tasks.oracle import get_true_image
from cliport.envs.tasks.random import Random


TRAIN_COLORS = ['blue', 'red', 'green', 'yellow', 'brown', 'gray', 'cyan']


COLORS = {
    'blue': [78.0 / 255.0, 121.0 / 255.0, 167.0 / 255.0],
    'red': [255.0 / 255.0, 87.0 / 255.0, 89.0 / 255.0],
    'green': [89.0 / 255.0, 169.0 / 255.0, 79.0 / 255.0],
    'orange': [242.0 / 255.0, 142.0 / 255.0, 43.0 / 255.0],
    'yellow': [237.0 / 255.0, 201.0 / 255.0, 72.0 / 255.0],
    'purple': [176.0 / 255.0, 122.0 / 255.0, 161.0 / 255.0],
    'pink': [255.0 / 255.0, 157.0 / 255.0, 167.0 / 255.0],
    'cyan': [118.0 / 255.0, 183.0 / 255.0, 178.0 / 255.0],
    'brown': [156.0 / 255.0, 117.0 / 255.0, 95.0 / 255.0],
    'white': [255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0],
    'gray': [186.0 / 255.0, 176.0 / 255.0, 172.0 / 255.0],
}


class PackingBoxesPairsSeenColors:
    """Packing Box Pairs task."""

    def __init__(self):
        self.max_steps = 20
        self.lang_template = "pack all the {colors} blocks into the brown box" # should have called it boxes :(
        self.task_completed_desc = "done packing blocks."

        # Tight z-bound (0.0525) to discourage stuffing everything into the brown box
        self.zone_bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.0525]])

        self.ee = Suction
        self.mode = 'train'
        self.sixdof = False
        self.primitive = PickPlace()

        # Evaluation epsilons (for pose evaluation metric).
        self.pos_eps = 0.01
        self.rot_eps = np.deg2rad(15)

        # Workspace bounds.
        self.pix_size = 0.003125
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.3]])
        self.zone_bounds = np.copy(self.bounds)

        self.goals = []
        self.lang_goals = []
        self.task_completed_desc = "task completed."
        self.progress = 0
        self._rewards = 0

        self.assets_root = "./cliport/envs/assets/"

    def done(self):
        return len(self.goals) == 0 or self._rewards > 0.99

    def reward(self):
        """Get delta rewards for current timestep.
        Returns:
            A tuple consisting of the scalar (delta) reward, plus `extras`
            dict which has extra task-dependent info from the process of
            computing rewards that gives us finer-grained details. Use
            `extras` for further data analysis.
        """
        reward, info = 0, {}

        # Unpack next goal step.
        objs, matches, targs, _, _, metric, params, max_reward = self.goals[0]

        # Evaluate by matching object poses.
        if metric == 'pose':
            step_reward = 0
            for i in range(len(objs)):
                object_id, (symmetry, _) = objs[i]
                pose = p.getBasePositionAndOrientation(object_id)
                targets_i = np.argwhere(matches[i, :]).reshape(-1)
                for j in targets_i:
                    target_pose = targs[j]
                    if self.is_match(pose, target_pose, symmetry):
                        step_reward += max_reward / len(objs)
                        break

        # Evaluate by measuring object intersection with zone.
        elif metric == 'zone':
            zone_pts, total_pts = 0, 0
            obj_pts, zones = params
            for zone_idx, (zone_pose, zone_size) in enumerate(zones):

                # Count valid points in zone.
                for obj_idx, obj_id in enumerate(obj_pts):
                    pts = obj_pts[obj_id]
                    obj_pose = p.getBasePositionAndOrientation(obj_id)
                    world_to_zone = Utils.invert(zone_pose)
                    obj_to_zone = Utils.multiply(world_to_zone, obj_pose)
                    pts = np.float32(Utils.apply(obj_to_zone, pts))
                    if len(zone_size) > 1:
                        valid_pts = np.logical_and.reduce([
                            pts[0, :] > -zone_size[0] / 2, pts[0, :] < zone_size[0] / 2,
                            pts[1, :] > -zone_size[1] / 2, pts[1, :] < zone_size[1] / 2,
                            pts[2, :] < self.zone_bounds[2, 1]])

                    # if zone_idx == matches[obj_idx].argmax():
                    zone_pts += np.sum(np.float32(valid_pts))
                    total_pts += pts.shape[1]
            step_reward = max_reward * (zone_pts / total_pts)

        # Get cumulative rewards and return delta.
        reward = self.progress + step_reward - self._rewards
        self._rewards = self.progress + step_reward

        # Move to next goal step if current goal step is complete.
        if np.abs(max_reward - step_reward) < 0.01:
            self.progress += max_reward  # Update task progress.
            self.goals.pop(0)
            if len(self.lang_goals) > 0:
                self.lang_goals.pop(0)

        return reward, info

    def get_lang_goal(self):
        if len(self.lang_goals) == 0:
            return "task completed."
        else:
            return self.lang_goals[0]

    def reset(self, env):
        if not self.assets_root:
            raise ValueError('assets_root must be set for task, call set_assets_root().')

        self.goals = []
        self.lang_goals = []
        self.progress = 0  # Task progression metric in range [0, 1].
        self._rewards = 0  # Cumulative returned rewards.

        # Add container box.
        zone_size = Random.get_random_size(0.05, 0.3, 0.05, 0.3, 0.05, 0.05)
        zone_pose = self.get_random_pose(env, zone_size)
        container_template = 'container/container-template.urdf'
        half = np.float32(zone_size) / 2
        replace = {'DIM': zone_size, 'HALF': half}
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, zone_pose, 'fixed')
        if os.path.exists(container_urdf):
            os.remove(container_urdf)

        margin = 0.01
        min_object_dim = 0.05
        bboxes = []

        class TreeNode:

            def __init__(self, parent, children, bbox):
                self.parent = parent
                self.children = children
                self.bbox = bbox  # min x, min y, min z, max x, max y, max z

        def KDTree(node):
            size = node.bbox[3:] - node.bbox[:3]

            # Choose which axis to split.
            split = size > 2 * min_object_dim
            if np.sum(split) == 0:
                bboxes.append(node.bbox)
                return
            split = np.float32(split) / np.sum(split)
            split_axis = np.random.choice(range(len(split)), 1, p=split)[0]

            # Split along chosen axis and create 2 children
            cut_ind = (
                np.random.rand()
                * (size[split_axis] - 2 * min_object_dim) 
                + node.bbox[split_axis] + min_object_dim
            )
            child1_bbox = node.bbox.copy()
            child1_bbox[3 + split_axis] = cut_ind - margin / 2.
            child2_bbox = node.bbox.copy()
            child2_bbox[split_axis] = cut_ind + margin / 2.
            node.children = [
                TreeNode(node, [], bbox=child1_bbox),
                TreeNode(node, [], bbox=child2_bbox)
            ]
            KDTree(node.children[0])
            KDTree(node.children[1])

        # Split container space with KD trees.
        stack_size = np.array(zone_size)
        stack_size[0] -= 0.01
        stack_size[1] -= 0.01
        root_size = (0.01, 0.01, 0) + tuple(stack_size)
        root = TreeNode(None, [], bbox=np.array(root_size))
        KDTree(root)

        all_color_names = [c for c in TRAIN_COLORS]
        relevant_color_names = np.random.choice(all_color_names, min(2, len(bboxes)), replace=False)
        distractor_color_names = [c for c in all_color_names if c not in relevant_color_names]

        pack_colors = [COLORS[c] for c in relevant_color_names]
        distractor_colors = [COLORS[c] for c in distractor_color_names]

        # Add objects in container.
        object_points = {}
        object_ids = []
        bboxes = np.array(bboxes)
        object_template = 'box/box-template.urdf'
        for bbox in bboxes:
            size = bbox[3:] - bbox[:3]
            position = size / 2. + bbox[:3]
            position[0] += -zone_size[0] / 2
            position[1] += -zone_size[1] / 2
            pose = (position, (0, 0, 0, 1))
            pose = Utils.multiply(zone_pose, pose)
            urdf = self.fill_template(object_template, {'DIM': size})
            box_id = env.add_object(urdf, pose)

            if os.path.exists(urdf):
                os.remove(urdf)

            object_ids.append((box_id, (0, None)))
            icolor = np.random.choice(range(len(pack_colors)), 1).squeeze()
            p.changeVisualShape(box_id, -1, rgbaColor=pack_colors[icolor] + [1])
            object_points[box_id] = self.get_box_object_points(box_id)

        # Randomly select object in box and save ground truth pose.
        object_volumes = []
        true_poses = []
        for object_id, _ in object_ids:
            true_pose = p.getBasePositionAndOrientation(object_id)
            object_size = p.getVisualShapeData(object_id)[0][3]
            object_volumes.append(np.prod(np.array(object_size) * 100))
            pose = self.get_random_pose(env, object_size)
            p.resetBasePositionAndOrientation(object_id, pose[0], pose[1])
            true_poses.append(true_pose)

        # Add distractor objects
        num_distractor_objects = 4
        distractor_bbox_idxs = np.random.choice(len(bboxes), num_distractor_objects)
        for bbox_idx in distractor_bbox_idxs:
            bbox = bboxes[bbox_idx]
            size = bbox[3:] - bbox[:3]
            position = size / 2. + bbox[:3]
            position[0] += -zone_size[0] / 2
            position[1] += -zone_size[1] / 2

            pose = self.get_random_pose(env, size)
            urdf = self.fill_template(object_template, {'DIM': size})
            box_id = env.add_object(urdf, pose)
            if os.path.exists(urdf):
                os.remove(urdf)
            icolor = np.random.choice(range(len(distractor_colors)), 1).squeeze()
            if box_id:
                p.changeVisualShape(box_id, -1, rgbaColor=distractor_colors[icolor] + [1])

        # Some scenes might contain just one relevant block that fits in the box.
        if len(relevant_color_names) > 1:
            relevant_desc = f'{relevant_color_names[0]} and {relevant_color_names[1]}'
        else:
            relevant_desc = f'{relevant_color_names[0]}'

        self.goals.append((
            object_ids,
            np.eye(len(object_ids)),
            true_poses,
            False,
            True,
            'zone',
            (object_points, [(zone_pose, zone_size)]),
            1
        ))
        self.lang_goals.append(self.lang_template.format(
            colors=relevant_desc,
        ))

    def get_random_pose(self, env, obj_size):
        """Get random collision-free object pose within workspace bounds."""

        # Get erosion size of object in pixels.
        max_size = np.sqrt(obj_size[0] ** 2 + obj_size[1] ** 2)
        erode_size = int(np.round(max_size / self.pix_size))

        _, hmap, obj_mask = get_true_image()

        # Randomly sample an object pose within free-space pixels.
        free = np.ones(obj_mask.shape, dtype=np.uint8)
        for obj_ids in env.obj_ids.values():
            for obj_id in obj_ids:
                free[obj_mask == obj_id] = 0
        free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
        free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
        if np.sum(free) == 0:
            return None, None
        pix = Random.sample_distribution(np.float32(free))
        pos = Transforms.pix_to_xyz(pix, hmap, self.bounds, self.pix_size)
        pos = (pos[0], pos[1], obj_size[2] / 2)
        theta = np.random.rand() * 2 * np.pi
        rot = Transforms.eulerXYZ_to_quatXYZW((0, 0, theta))
        return pos, rot

    def is_match(self, pose0, pose1, symmetry):
        """Check if pose0 and pose1 match within a threshold."""

        # Get translational error.
        diff_pos = np.float32(pose0[0][:2]) - np.float32(pose1[0][:2])
        dist_pos = np.linalg.norm(diff_pos)

        # Get rotational error around z-axis (account for symmetries).
        diff_rot = 0
        if symmetry > 0:
            rot0 = np.array(Transforms.quatXYZW_to_eulerXYZ(pose0[1]))[2]
            rot1 = np.array(Transforms.quatXYZW_to_eulerXYZ(pose1[1]))[2]
            diff_rot = np.abs(rot0 - rot1) % symmetry
            if diff_rot > (symmetry / 2):
                diff_rot = symmetry - diff_rot

        return (dist_pos < self.pos_eps) and (diff_rot < self.rot_eps)

    def oracle(self, env):
        """Oracle agent."""
        OracleAgent = collections.namedtuple('OracleAgent', ['act'])

        def act(obs, info):  # pylint: disable=unused-argument
            """Calculate action."""

            # Oracle uses perfect RGB-D orthographic images and segmentation masks.
            _, hmap, obj_mask = get_true_image()

            # Unpack next goal step.
            objs, matches, targs, replace, rotations, _, _, _ = self.goals[0]

            # Match objects to targets without replacement.
            if not replace:

                # Modify a copy of the match matrix.
                matches = matches.copy()

                # Ignore already matched objects.
                for i in range(len(objs)):
                    object_id, (symmetry, _) = objs[i]
                    pose = p.getBasePositionAndOrientation(object_id)
                    targets_i = np.argwhere(matches[i, :]).reshape(-1)
                    for j in targets_i:
                        if self.is_match(pose, targs[j], symmetry):
                            matches[i, :] = 0
                            matches[:, j] = 0

            # Get objects to be picked (prioritize farthest from nearest neighbor).
            nn_dists = []
            nn_targets = []
            for i in range(len(objs)):
                object_id, (symmetry, _) = objs[i]
                xyz, _ = p.getBasePositionAndOrientation(object_id)
                targets_i = np.argwhere(matches[i, :]).reshape(-1)
                if len(targets_i) > 0:  # pylint: disable=g-explicit-length-test
                    targets_xyz = np.float32([targs[j][0] for j in targets_i])
                    dists = np.linalg.norm(
                        targets_xyz - np.float32(xyz).reshape(1, 3), axis=1)
                    nn = np.argmin(dists)
                    nn_dists.append(dists[nn])
                    nn_targets.append(targets_i[nn])

                # Handle ignored objects.
                else:
                    nn_dists.append(0)
                    nn_targets.append(-1)

            order = np.argsort(nn_dists)[::-1]

            # Filter out matched objects.
            order = [i for i in order if nn_dists[i] > 0]

            pick_mask = None
            for pick_i in order:
                pick_mask = np.uint8(obj_mask == objs[pick_i][0])

                # Erode to avoid picking on edges.
                # pick_mask = cv2.erode(pick_mask, np.ones((3, 3), np.uint8))

                if np.sum(pick_mask) > 0:
                    break

            # Trigger task reset if no object is visible.
            if pick_mask is None or np.sum(pick_mask) == 0:
                self.goals = []
                self.lang_goals = []
                print('Object for pick is not visible. Skipping demonstration.')
                return

            # Get picking pose.
            pick_prob = np.float32(pick_mask)
            pick_pix = Random.sample_distribution(pick_prob)
            # For "deterministic" demonstrations on insertion-easy, use this:
            # pick_pix = (160,80)
            pick_pos = Transforms.pix_to_xyz(
                pick_pix,
                hmap,
                self.bounds,
                self.pix_size
            )
            pick_pose = (np.asarray(pick_pos), np.asarray((0, 0, 0, 1)))

            # Get placing pose.
            targ_pose = targs[nn_targets[pick_i]]  # pylint: disable=undefined-loop-variable
            obj_pose = p.getBasePositionAndOrientation(objs[pick_i][0])  # pylint: disable=undefined-loop-variable

            if not self.sixdof:
                obj_euler = Transforms.quatXYZW_to_eulerXYZ(obj_pose[1])
                obj_quat = Transforms.eulerXYZ_to_quatXYZW((0, 0, obj_euler[2]))
                obj_pose = (obj_pose[0], obj_quat)

            world_to_pick = Utils.invert(pick_pose)
            obj_to_pick = Utils.multiply(world_to_pick, obj_pose)
            pick_to_obj = Utils.invert(obj_to_pick)
            place_pose = Utils.multiply(targ_pose, pick_to_obj)

            # Rotate end effector?
            if not rotations:
                place_pose = (place_pose[0], (0, 0, 0, 1))

            place_pose = (np.asarray(place_pose[0]), np.asarray(place_pose[1]))

            return {'pose0': pick_pose, 'pose1': place_pose}

        return OracleAgent(act)

    def fill_template(self, template, replace):
        """Read a file and replace key strings."""
        full_template_path = os.path.join(self.assets_root, template)
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

    def get_box_object_points(self, obj):
        obj_shape = p.getVisualShapeData(obj)
        obj_dim = obj_shape[0][3]
        obj_dim = tuple(d for d in obj_dim)
        xv, yv, zv = np.meshgrid(
            np.arange(-obj_dim[0] / 2, obj_dim[0] / 2, 0.02),
            np.arange(-obj_dim[1] / 2, obj_dim[1] / 2, 0.02),
            np.arange(-obj_dim[2] / 2, obj_dim[2] / 2, 0.02),
            sparse=False, indexing='xy')
        return np.vstack((xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)))
