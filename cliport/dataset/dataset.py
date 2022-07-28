import os
import pickle
import warnings
import numpy as np

from torch.utils.data import Dataset

from cliport.dataset.utils import (
    get_fused_heightmap,
    xyz_to_pix,
    quatXYZW_to_eulerXYZ,
    perturb,
    apply_perturbation,
)
from cliport.dataset.cameras import RealSenseD415


class RavensDataset(Dataset):
    """A simple image dataset class."""

    def __init__(self, path, n_demos=0):
        """A simple RGB-D image dataset."""

        # task data path
        self._path = path
        self.sample_set = []
        self.max_seed = -1
        self.n_episodes = 0
        self.images = True
        self.cache = True
        self.n_demos = n_demos

        self.aug_theta_sigma = 60  # legacy code issue: theta_sigma was newly added
        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        # Track existing dataset if it exists.
        color_path = os.path.join(self._path, 'action')
        if os.path.exists(color_path):
            for fname in sorted(os.listdir(color_path)):
                if '.pkl' in fname:
                    seed = int(fname[(fname.find('-') + 1):-4])
                    self.n_episodes += 1
                    self.max_seed = max(self.max_seed, seed)

        self._cache = {}

        if self.n_demos > 0:
            episodes = np.random.choice(
                range(self.n_episodes),
                self.n_demos,
                False
            )
            self.set(episodes)

    def add(self, seed, episode):
        """Add an episode to the dataset.
        Args:
            seed: random seed used to initialize the episode.
            episode: list of (obs, act, reward, info) tuples.
        """
        color, depth, action, reward, info = [], [], [], [], []
        for obs, act, r, i in episode:
            color.append(obs['color'])
            depth.append(obs['depth'])
            action.append(act)
            reward.append(r)
            info.append(i)

        color = np.uint8(color)
        depth = np.float32(depth)

        def dump(data, field):
            field_path = os.path.join(self._path, field)
            if not os.path.exists(field_path):
                os.makedirs(field_path)
            fname = f'{self.n_episodes:06d}-{seed}.pkl'  # -{len(episode):06d}
            with open(os.path.join(field_path, fname), 'wb') as f:
                pickle.dump(data, f)

        dump(color, 'color')
        dump(depth, 'depth')
        dump(action, 'action')
        dump(reward, 'reward')
        dump(info, 'info')

        self.n_episodes += 1
        self.max_seed = max(self.max_seed, seed)

    def set(self, episodes):
        """Limit random samples to specific fixed set."""
        self.sample_set = episodes

    def load(self, episode_id, images=True):

        def load_field(episode_id, field, fname):
            # Check if sample is in cache.
            if self.cache:
                if episode_id in self._cache:
                    if field in self._cache[episode_id]:
                        return self._cache[episode_id][field]
                else:
                    self._cache[episode_id] = {}

            # Load sample from files.
            path = os.path.join(self._path, field)
            data = pickle.load(open(os.path.join(path, fname), 'rb'))

            if self.cache:
                self._cache[episode_id][field] = data

            return data

        # Get filename and random seed used to initialize episode.
        seed = None
        path = os.path.join(self._path, 'action')
        for fname in sorted(os.listdir(path)):
            if f'{episode_id:06d}' in fname:
                seed = int(fname[(fname.find('-') + 1):-4])

                # Load data.
                color = load_field(episode_id, 'color', fname)
                depth = load_field(episode_id, 'depth', fname)
                action = load_field(episode_id, 'action', fname)
                reward = load_field(episode_id, 'reward', fname)
                info = load_field(episode_id, 'info', fname)

                # Reconstruct episode.
                episode = []
                for i in range(len(action)):
                    obs = {'color': color[i], 'depth': depth[i]} if images else {}
                    episode.append((obs, action[i], reward[i], info[i]))
                return episode, seed

    def get_image(self, obs):
        """Stack color and height images image."""

        # Get color and height maps from RGB-D images.
        cmap, hmap = get_fused_heightmap(
            obs,
            RealSenseD415.CONFIG,
            self.bounds,
            self.pix_size
        )
        img = np.concatenate((
            cmap,
            hmap[Ellipsis, None],
            hmap[Ellipsis, None],
            hmap[Ellipsis, None]
        ), axis=2)

        assert img.shape == self.in_shape, img.shape
        return img

    def process_sample(self, datum):
        # Get training labels from data sample.
        (obs, act, _, info) = datum
        img = self.get_image(obs)

        p0, p1 = None, None
        p0_theta, p1_theta = None, None
        perturb_params =  None

        if act:
            p0_xyz, p0_xyzw = act['pose0']
            p1_xyz, p1_xyzw = act['pose1']
            p0 = xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
            p0_theta = -np.float32(quatXYZW_to_eulerXYZ(p0_xyzw)[2])
            p1 = xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
            p1_theta = -np.float32(quatXYZW_to_eulerXYZ(p1_xyzw)[2])
            p1_theta = p1_theta - p0_theta
            p0_theta = 0

        # Data augmentation.
        img, _, (p0, p1), perturb_params = perturb(
            img,
            [p0, p1],
            theta_sigma=self.aug_theta_sigma
        )

        sample = {
            'img': img,
            'p0': p0,
            'p0_theta': p0_theta,
            'p1': p1,
            'p1_theta': p1_theta,
            'perturb_params': perturb_params
        }

        # Add language goal if available.
        if 'lang_goal' not in info:
            warnings.warn("No language goal. Defaulting to 'task completed.'")

        if info and 'lang_goal' in info:
            sample['lang_goal'] = info['lang_goal']
        else:
            sample['lang_goal'] = "task completed."

        return sample

    def process_goal(self, goal, perturb_params):
        # Get goal sample.
        (obs, act, _, info) = goal
        img = self.get_image(obs)

        p0, p1 = None, None
        p0_theta, p1_theta = None, None

        # Data augmentation with specific params.
        if perturb_params:
            img = apply_perturbation(img, perturb_params)

        sample = {
            'img': img,
            'p0': p0,
            'p0_theta': p0_theta,
            'p1': p1,
            'p1_theta': p1_theta,
            'perturb_params': perturb_params
        }

        # Add language goal if available.
        if 'lang_goal' not in info:
            warnings.warn("No language goal. Defaulting to 'task completed.'")

        if info and 'lang_goal' in info:
            sample['lang_goal'] = info['lang_goal']
        else:
            sample['lang_goal'] = "task completed."

        return sample

    def __len__(self):
        return len(self.sample_set)

    def __getitem__(self, idx):

        # Choose random episode.
        if len(self.sample_set) > 0:
            episode_id = np.random.choice(self.sample_set)
        else:
            episode_id = np.random.choice(range(self.n_episodes))

        episode, _ = self.load(episode_id, self.images)

        # Is the task sequential like stack-block-pyramid-seq?
        is_sequential_task = '-seq' in self._path.split("/")[-1]

        # Return random observation action pair (and goal) from episode.
        i = np.random.choice(range(len(episode)-1))
        g = i + 1 if is_sequential_task else -1
        sample, goal = episode[i], episode[g]

        # Process sample.
        sample = self.process_sample(sample)
        goal = self.process_goal(
            goal,
            perturb_params=sample['perturb_params']
        )

        return sample, goal
