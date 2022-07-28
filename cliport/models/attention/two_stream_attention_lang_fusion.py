import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from cliport.models.attention.clip_lingunet_lat import CLIPLingUNetLat
from cliport.models.network.fusion import FusionAdd
from cliport.models.network.resnet_lat import ResNet45_10s
from cliport.models.utils import ImageRotator


class TwoStreamAttentionLangFusionLat(nn.Module):
    """Two Stream Language-Conditioned Attention (a.k.a Pick) module."""

    def __init__(self, in_shape, n_rotations, device):
        super().__init__()
        self.n_rotations = n_rotations
        self.device = device

        self.padding = np.zeros((3, 2), dtype=int)
        max_dim = np.max(in_shape[:2])
        pad = (max_dim - np.array(in_shape[:2])) / 2
        self.padding[:2] = pad.reshape(2, 1)

        in_shape = np.array(in_shape)
        in_shape += np.sum(self.padding, axis=1)
        in_shape = tuple(in_shape)
        self.in_shape = in_shape

        self.rotator = ImageRotator(self.n_rotations)

        self.attn_stream_one = ResNet45_10s(
            self.in_shape,
            1,
            self.device,
        )
        self.attn_stream_two = CLIPLingUNetLat(
            self.in_shape,
            1,
            self.device,
        )
        self.fusion = FusionAdd(input_dim=1)

    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        in_data = np.pad(inp_img, self.padding, mode='constant')
        in_shape = (1,) + in_data.shape
        in_data = in_data.reshape(in_shape)
        in_tens = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)  # [B W H 6]

        # Rotation pivot.
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        in_tens = in_tens.permute(0, 3, 1, 2)  # [B 6 W H]
        in_tens = in_tens.repeat(self.n_rotations, 1, 1, 1)
        in_tens = self.rotator(in_tens, pivot=pv)

        # Forward pass.
        logits = []
        for x in in_tens:
            lgts = self.attend(x, lang_goal)
            logits.append(lgts)
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]

        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))

        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])

        return output

    def attend(self, x, l):
        x1, lat = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, lat, l)
        x = self.fusion(x1, x2)
        return x
