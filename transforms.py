import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def upsample(image, image_size_target):
    padding0 = (image_size_target - image.shape[0]) / 2
    padding1 = (image_size_target - image.shape[1]) / 2
    padding_start0 = int(np.ceil(padding0))
    padding_end0 = int(np.floor(padding0))
    padding_start1 = int(np.ceil(padding1))
    padding_end1 = int(np.floor(padding1))
    return cv2.copyMakeBorder(image, padding_start0, padding_end0, padding_start1, padding_end1, cv2.BORDER_REFLECT_101)


def downsample(image, image_size_original):
    padding = (image.shape[0] - image_size_original) / 2
    padding_start = int(np.ceil(padding))
    return image[padding_start:padding_start + image_size_original, padding_start:padding_start + image_size_original]


def augment(image, mask):
    if np.random.rand() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    if np.random.rand() < 0.5:
        c = np.random.choice(2)
        if c == 0:
            image = multiply_brightness(image, np.random.uniform(1 - 0.1, 1 + 0.1))
        elif c == 1:
            image = adjust_gamma(image, np.random.uniform(1 - 0.1, 1 + 0.1))

    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c == 0:
            image, mask = apply_elastic_transform(image, mask, alpha=150, sigma=8, alpha_affine=0)
        elif c == 1:
            image, mask = apply_elastic_transform(image, mask, alpha=0, sigma=0, alpha_affine=8)
        elif c == 2:
            image, mask = apply_elastic_transform(image, mask, alpha=150, sigma=10, alpha_affine=5)

    if np.random.rand() < 0.5:
        image, mask = random_crop_and_pad(image, mask)

    return image, mask


def multiply_brightness(image, coefficient):
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image_HLS = np.array(image_HLS, dtype=np.float64)
    image_HLS[:, :, 1] = image_HLS[:, :, 1] * coefficient
    image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
    image_HLS = np.array(image_HLS, dtype=np.uint8)
    return cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)


def adjust_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)


def apply_elastic_transform(image, mask, alpha, sigma, alpha_affine):
    channels = np.concatenate((image, mask[..., None]), axis=2)
    result = elastic_transform(channels, alpha, sigma, alpha_affine, random_state=np.random.RandomState(None))
    image_result = result[..., 0:3]
    mask_result = result[..., 3]
    mask_result = (mask_result > 0.5).astype(mask.dtype)
    return image_result, mask_result


def random_crop_and_pad(image, mask):
    max_crop = 40

    crop_x_total = np.random.randint(max_crop)
    crop_x0 = np.random.randint(crop_x_total + 1)
    crop_x1 = crop_x_total - crop_x0

    crop_y_total = np.random.randint(max_crop)
    crop_y0 = np.random.randint(crop_y_total + 1)
    crop_y1 = crop_y_total - crop_y0

    cropped_image = image[crop_x0:image.shape[0] - crop_x1, crop_y0:image.shape[1] - crop_y1, :]
    cropped_mask = mask[crop_x0:mask.shape[0] - crop_x1, crop_y0:mask.shape[1] - crop_y1]

    cropped_padded_image = upsample(cropped_image, image.shape[0])
    cropped_padded_mask = upsample(cropped_mask, mask.shape[0])

    return cropped_padded_image, cropped_padded_mask


def random_crop_to_size(image, mask, size):
    dmax = image.shape[0] - size

    dx = np.random.randint(dmax + 1)
    dy = np.random.randint(dmax + 1)

    cropped_image = image[dx:dx + size, dy:dy + size, :]
    cropped_mask = mask[dx:dx + size, dy:dy + size]

    return cropped_image, cropped_mask
