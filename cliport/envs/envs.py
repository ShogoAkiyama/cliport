"""Environment class."""

import os
import tempfile
import time
import gym
import numpy as np
import pybullet as p

from cliport.dataset.cameras import RealSenseD415
from cliport.envs.tasks.packing_boxes_pairs import PackingBoxesPairsSeenColors
from cliport.envs.utils import load_urdf, render_camera
from cliport.envs.tasks.grippers import Suction

UR5_URDF_PATH = 'ur5/ur5.urdf'
UR5_WORKSPACE_URDF_PATH = 'ur5/workspace.urdf'
PLANE_URDF_PATH = 'plane/plane.urdf'


class Environment(gym.Env):
    """OpenAI Gym-style environment class."""

    def __init__(self):
        self.assets_root = "./cliport/envs/assets/"
        hz = 480
        self.record_cfg = {
            "fps": 20,
            "video_height": 640,
            "video_width": 720,
        }
        self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.agent_cams = RealSenseD415.CONFIG
        self.step_counter = 0

        color_tuple = [
            gym.spaces.Box(0, 255, config['image_size'] + (3,), dtype=np.uint8)
            for config in self.agent_cams
        ]
        depth_tuple = [
            gym.spaces.Box(0.0, 20.0, config['image_size'], dtype=np.float32)
            for config in self.agent_cams
        ]

        self.observation_space = gym.spaces.Dict({
            'color': gym.spaces.Tuple(color_tuple),
            'depth': gym.spaces.Tuple(depth_tuple),
        })

        self.position_bounds = gym.spaces.Box(
            low=np.array([0.25, -0.5, 0.], dtype=np.float32),
            high=np.array([0.75, 0.5, 0.28], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Dict({
            'pose0':
                gym.spaces.Tuple((
                    self.position_bounds,
                    gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
                )),
            'pose1':
                gym.spaces.Tuple((
                    self.position_bounds,
                    gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
                ))
        })

        # Start PyBullet.
        client = p.connect(p.DIRECT)
        file_io = p.loadPlugin('fileIOPlugin', physicsClientId=client)
        if file_io < 0:
            raise RuntimeError('pybullet: cannot load FileIO!')

        if file_io >= 0:
            p.executePluginCommand(
                file_io,
                textArgument=self.assets_root,
                intArgs=[p.AddFileIOAction],
                physicsClientId=client
            )

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setAdditionalSearchPath(self.assets_root)
        p.setAdditionalSearchPath(tempfile.gettempdir())
        p.setTimeStep(1. / hz)

        self.task = PackingBoxesPairsSeenColors()

    def reset(self):
        """Performs common reset functionality for all supported tasks."""
        if not self.task:
            raise ValueError(
                'environment task must be set. Call set_task or pass '
                'the task arg in the environment constructor.'
            )

        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        load_urdf(
            p,
            os.path.join(self.assets_root, PLANE_URDF_PATH),
            [0, 0, -0.001]
        )
        load_urdf(
            p,
            os.path.join(self.assets_root, UR5_WORKSPACE_URDF_PATH),
            [0.5, 0, 0]
        )

        # Reset task.
        self.task.reset()

        # Load UR5 robot arm equipped with suction end effector.
        # TODO(andyzeng): add back parallel-jaw grippers.
        self.ur5 = load_urdf(
            p,
            os.path.join(self.assets_root, UR5_URDF_PATH)
        )

        assert self.ur5 is not None

        self.ee = Suction(
            self.assets_root,
            self.ur5,
            9,
            self.task.obj_ids
        )
        self.ee_tip = 10  # Link ID of suction cup.

        # Reset end effector.
        self.ee.release()

        # Get revolute joint indices of robot (skip fixed joints).
        n_joints = p.getNumJoints(self.ur5)
        joints = [p.getJointInfo(self.ur5, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        # Move robot to home joint configuration.
        for i in range(len(self.joints)):
            p.resetJointState(self.ur5, self.joints[i], self.homej[i])

        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        obs, _, _, _ = self.step()

        return obs

    def step(self, action=None):

        if action is not None:
            timeout = self.task.primitive(
                self.movep,
                self.ee,
                action['pose0'],
                action['pose1']
            )

            # Exit early if action times out. We still return an observation
            # so that we don't break the Gym API contract.
            if timeout:
                obs = {'color': (), 'depth': ()}
                for config in self.agent_cams:
                    color, depth, _ = self.render_camera(config)
                    obs['color'] += (color,)
                    obs['depth'] += (depth,)
                return obs, 0.0, True, self.info

        # Step simulator asynchronously until objects settle.
        while not self.is_static:
            self.step_simulation()

        # Get task rewards.
        reward, info = self.task.reward() if action is not None else (0, {})
        done = self.task.done()

        # Add ground truth robot state into info.
        info.update(self.info)

        obs = self._get_obs()

        return obs, reward, done, info

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [
            np.linalg.norm(p.getBaseVelocity(i)[0])
            for i in self.task.obj_ids['rigid']
        ]
        return all(np.array(v) < 5e-3)

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def step_simulation(self):
        p.stepSimulation()
        self.step_counter += 1

    def render(self, mode='rgb_array'):
        if mode != 'rgb_array':
            raise NotImplementedError('Only rgb_array implemented')
        color, _, _ = self.render_camera(self.agent_cams[0])
        return color

    @property
    def info(self):
        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.task.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = p.getBasePositionAndOrientation(obj_id)
                dim = p.getVisualShapeData(obj_id)[0][3]
                info[obj_id] = (pos, rot, dim)

        info['lang_goal'] = self.get_lang_goal()
        return info

    def get_lang_goal(self):
        if self.task:
            return self.task.get_lang_goal()
        else:
            raise Exception("No task for was set")

    def movej(self, targj, speed=0.01, timeout=5):
        t0 = time.time()
        while (time.time() - t0) < timeout:
            currj = [p.getJointState(self.ur5, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return False

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.ur5,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains
            )
            self.step_counter += 1
            self.step_simulation()

        print(f'Warning: movej exceeded {timeout} second timeout. Skipping.')
        return True

    def movep(self, pose, speed=0.01):
        """Move UR5 to target end effector pose."""
        targj = self.solve_ik(pose)
        return self.movej(targj, speed)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = p.calculateInverseKinematics(
            bodyUniqueId=self.ur5,
            endEffectorLinkIndex=self.ee_tip,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-3 * np.pi / 2, -2.3562, -17, -17, -17, -17],
            upperLimits=[-np.pi / 2, 0, 17, 17, 17, 17],
            jointRanges=[np.pi, 2.3562, 34, 34, 34, 34],  # * 6,
            restPoses=np.float32(self.homej).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        joints = np.float32(joints)
        joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def _get_obs(self):
        # Get RGB-D camera image observations.
        obs = {
            'color': (),
            'depth': ()
        }
        for config in self.agent_cams:
            color, depth, _ = render_camera(config)
            obs['color'] += (color,)
            obs['depth'] += (depth,)

        return obs
