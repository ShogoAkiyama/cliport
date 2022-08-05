import numpy as np
import cv2

from cliport.dataset.transform import Transform


def perturb(input_image, pixels, theta_sigma=60, add_noise=False):
    """Data augmentation on images."""
    image_size = input_image.shape[:2]

    # Compute random rigid transform.
    while True:
        theta, trans, pivot = Transform.get_random_image_transform_params(
            image_size,
            theta_sigma=theta_sigma
        )
        transform = Transform.get_image_transform(theta, trans, pivot)
        transform_params = theta, trans, pivot

        # Ensure pixels remain in the image after transform.
        is_valid = True
        new_pixels = []
        new_rounded_pixels = []
        for pixel in pixels:
            pixel = np.float32([pixel[1], pixel[0], 1.]).reshape(3, 1)

            rounded_pixel = np.int32(np.round(transform @ pixel))[:2].squeeze()
            rounded_pixel = np.flip(rounded_pixel)

            pixel = (transform @ pixel)[:2].squeeze()
            pixel = np.flip(pixel)

            in_fov_rounded = rounded_pixel[0] < image_size[0] and rounded_pixel[
                1] < image_size[1]
            in_fov = pixel[0] < image_size[0] and pixel[1] < image_size[1]

            is_valid = is_valid and np.all(rounded_pixel >= 0) and np.all(
                pixel >= 0) and in_fov_rounded and in_fov

            new_pixels.append(pixel)
            new_rounded_pixels.append(rounded_pixel)
        if is_valid:
            break

    # Apply rigid transform to image and pixel labels.
    input_image = cv2.warpAffine(
        input_image,
        transform[:2, :], (image_size[1], image_size[0]),
        flags=cv2.INTER_LINEAR)

    # Apply noise
    color = np.int32(input_image[:,:,:3])
    depth = np.float32(input_image[:,:,3:])

    if add_noise:
        color += np.int32(np.random.normal(0, 3, image_size + (3,)))
        color = np.uint8(np.clip(color, 0, 255))

        depth += np.float32(np.random.normal(0, 0.003, image_size + (3,)))

    input_image = np.concatenate((color, depth), axis=2)

    return input_image, new_pixels, new_rounded_pixels, transform_params


def apply_perturbation(input_image, transform_params):
    '''Apply data augmentation with specific transform params'''
    image_size = input_image.shape[:2]

    # Apply rigid transform to image and pixel labels.
    theta, trans, pivot = transform_params
    transform = Transform.get_image_transform(theta, trans, pivot)

    input_image = cv2.warpAffine(
        input_image,
        transform[:2, :], (image_size[1], image_size[0]),
        flags=cv2.INTER_LINEAR)
    return input_image
