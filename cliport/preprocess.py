import numpy as np
import torch


def preprocess(img):
    """Pre-process input (subtract mean, divide by std)."""

    transporter_color_mean = [0.18877631, 0.18877631, 0.18877631]
    transporter_color_std = [0.07276466, 0.07276466, 0.07276466]
    transporter_depth_mean = 0.00509261
    transporter_depth_std = 0.00903967

    # choose distribution
    color_mean = transporter_color_mean
    color_std = transporter_color_std

    depth_mean = transporter_depth_mean
    depth_std = transporter_depth_std

    # convert to pytorch tensor (if required)
    if type(img) == torch.Tensor:
        def cast_shape(stat, img):
            tensor = torch.from_numpy(np.array(stat)).to(device=img.device, dtype=img.dtype)
            tensor = tensor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            tensor = tensor.repeat(img.shape[0], 1, img.shape[-2], img.shape[-1])
            return tensor

        color_mean = cast_shape(color_mean, img)
        color_std = cast_shape(color_std, img)
        depth_mean = cast_shape(depth_mean, img)
        depth_std = cast_shape(depth_std, img)

        # normalize
        img = img.clone()
        img[:, :3, :, :] = ((img[:, :3, :, :] / 255 - color_mean) / color_std)
        img[:, 3:, :, :] = ((img[:, 3:, :, :] - depth_mean) / depth_std)
    else:
        # normalize
        img[:, :, :3] = (img[:, :, :3] / 255 - color_mean) / color_std
        img[:, :, 3:] = (img[:, :, 3:] - depth_mean) / depth_std

    return img
