import numpy as np

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from cliport.utils import set_seed
from cliport.models.transport.two_stream_transport_lang_fusion import TwoStreamTransportLangFusionLat
from cliport.models.attention.two_stream_attention_lang_fusion import TwoStreamAttentionLangFusionLat
from cliport.envs.tasks.transform import Transforms


class TwoStreamClipLingUNetLatTransporterAgent(LightningModule):
    def __init__(self, train_ds, test_ds):
        super().__init__()
        set_seed(0)

        self.device_type = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.train_ds = train_ds
        self.test_ds = test_ds

        self.total_steps = 0
        self.crop_size = 64
        self.n_rotations = 36
        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
        self.val_repeats = 1

        # build model
        self.attention = TwoStreamAttentionLangFusionLat(
            in_shape=self.in_shape,
            n_rotations=1,
            device=self.device_type,
        )
        self.transport = TwoStreamTransportLangFusionLat(
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            device=self.device_type,
        )

        self.attn_optim = torch.optim.Adam(
            self.attention.parameters(),
            lr=1e-4
        )
        self.trans_optim = torch.optim.Adam(
            self.transport.parameters(),
            lr=1e-4
        )

    def training_step(self, batch):
        self.attention.train()
        self.transport.train()

        frame, _ = batch

        # Get training losses.
        step = self.total_steps + 1
        self.attn_training_step(frame)
        self.transport_training_step(frame)

        self.total_steps = step

    def attn_training_step(self, frame, backprop=True):
        inp_img = frame['img']
        p0 = frame['p0']
        p0_theta = frame['p0_theta']
        lang_goal = frame['lang_goal']

        out = self.attention.forward(
            inp_img,
            lang_goal,
            softmax=False
        )

        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}

        self.attn_criterion(backprop, inp, out, p0, p0_theta)

    def transport_training_step(self, frame, backprop=True):
        inp_img = frame['img']
        p0 = frame['p0']
        p1 = frame['p1']
        p1_theta = frame['p1_theta']
        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'p0': p0, 'lang_goal': lang_goal}

        out = self.transport.forward(
            inp_img,
            p0,
            lang_goal,
            softmax=False
        )

        self.transport_criterion(backprop, inp, out, p1, p1_theta)

    def attn_criterion(self, backprop, inp, out, p, theta):
        # Get label.
        theta_i = theta / (2 * np.pi / self.attention.n_rotations)
        theta_i = np.int32(np.round(theta_i)) % self.attention.n_rotations
        label_size = inp['inp_img'].shape[:2] + (self.attention.n_rotations,)
        label = np.zeros(label_size)
        label[p[0], p[1], theta_i] = 1
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=out.device)

        # Get loss.
        loss = cross_entropy_with_logits(out, label)

        # Backpropagate.
        if backprop:
            self.manual_backward(
                loss,
                self.attn_optim,
            )
            self.attn_optim.step()
            self.attn_optim.zero_grad()

    def transport_criterion(self, backprop, inp, output, q, theta):
        itheta = theta / (2 * np.pi / self.transport.n_rotations)
        itheta = np.int32(np.round(itheta)) % self.transport.n_rotations

        # Get one-hot pixel label map.
        label_size = inp['inp_img'].shape[:2] + (self.transport.n_rotations,)
        label = np.zeros(label_size)
        label[q[0], q[1], itheta] = 1

        # Get loss.
        label = label.transpose((2, 0, 1))
        label = label.reshape(1, np.prod(label.shape))
        label = torch.from_numpy(label).to(dtype=torch.float, device=output.device)
        output = output.reshape(1, np.prod(output.shape))

        loss = cross_entropy_with_logits(output, label)

        if backprop:
            self.manual_backward(
                loss,
                self.trans_optim
            )
            self.trans_optim.step()
            self.trans_optim.zero_grad()

        self.transport.iters += 1

    def act(self, obs, info):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        img = self.test_ds.get_image(obs)
        lang_goal = info['lang_goal']

        # pick_conf = self.attn_forward(pick_inp)
        pick_conf = self.attention.forward(
            img,
            lang_goal,
            softmax=True
        )
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Transport model forward pass.
        place_conf = self.transport.forward(
            img,
            p0_pix,
            lang_goal,
            softmax=True,
        )
        place_conf = place_conf.permute(1, 2, 0)
        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / place_conf.shape[2])

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        p0_xyz = Transforms.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        p1_xyz = Transforms.pix_to_xyz(p1_pix, hmap, self.bounds, self.pix_size)
        p0_xyzw = Transforms.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))
        p1_xyzw = Transforms.eulerXYZ_to_quatXYZW((0, 0, -p1_theta))

        return {
            'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pose1': (np.asarray(p1_xyz), np.asarray(p1_xyzw)),
            'pick': [p0_pix[0], p0_pix[1], p0_theta],
            'place': [p1_pix[0], p1_pix[1], p1_theta],
        }


def cross_entropy_with_logits(pred, labels):
    # Lucas found that both sum and mean work equally well
    x = (-labels * F.log_softmax(pred, -1))
    return x.mean()
