import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from cliport.models.attention.clip_lingunet_lat import CLIPLingUNetLat
from cliport.models.network.fusion import FusionAdd
from cliport.models.network.resnet_lat import ResNet45_10s
from cliport.models.utils import ImageRotator


class TwoStreamTransportLangFusionLat(nn.Module):
    """Two Stream Transport (a.k.a Place) module"""

    def __init__(self, in_shape, n_rotations, crop_size, device):
        super().__init__()
        self.n_rotations = n_rotations
        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
        self.device = device

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

        self.in_shape = tuple(np.array(in_shape))

        # Crop before network (default from Transporters CoRL 2020).
        self.kernel_shape = (
            self.crop_size,
            self.crop_size,
            self.in_shape[2]
        )

        if not hasattr(self, 'output_dim'):
            self.output_dim = 3
        if not hasattr(self, 'kernel_dim'):
            self.kernel_dim = 3

        self.rotator = ImageRotator(self.n_rotations)

        self.key_stream_one = ResNet45_10s(
            self.in_shape,
            self.output_dim,
            self.device,
        )
        self.key_stream_two = CLIPLingUNetLat(
            self.in_shape,
            self.output_dim,
            self.device,
        )
        self.query_stream_one = ResNet45_10s(
            self.kernel_shape,
            self.kernel_dim,
            self.device,
        )
        self.query_stream_two = CLIPLingUNetLat(
            self.kernel_shape,
            self.kernel_dim,
            self.device,
        )
        self.fusion_key = FusionAdd(input_dim=self.kernel_dim)
        self.fusion_query = FusionAdd(input_dim=self.kernel_dim)

    def forward(self, inp_img, p, lang_goal, softmax=True):
        """Forward pass."""
        input_data = np.pad(inp_img, self.padding, mode='constant')
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape)
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)

        # Rotation pivot.
        pv = np.array([p[0], p[1]]) + self.pad_size

        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2)

        crop = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        crop = self.rotator(crop, pivot=pv)
        crop = torch.cat(crop, dim=0)
        crop = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        logits, kernel = self.transport(in_tensor, crop, lang_goal)

        return self.correlate(logits, kernel, softmax)

    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        output = F.conv2d(in0, in1, padding=(self.pad_size, self.pad_size))
        output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
        output = output[:, :, self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]

        if softmax:
            output_shape = output.shape
            output = output.reshape((1, np.prod(output.shape)))
            output = F.softmax(output, dim=-1)
            output = output.reshape(output_shape[1:])

        return output

    def transport(self, in_tensor, crop, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)
        logits = self.fusion_key(key_out_one, key_out_two)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)
        kernel = self.fusion_query(query_out_one, query_out_two)

        return logits, kernel
