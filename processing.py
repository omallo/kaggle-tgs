from multiprocessing import Pool

import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from scipy import ndimage
from skimage.color import gray2rgb


def calculate_contour(mask, width=3):
    edge_x = ndimage.convolve(mask, np.array([[-1, 0, +1], [-1, 0, +1], [-1, 0, +1]]))
    edge_y = ndimage.convolve(mask, np.array([[-1, -1, -1], [0, 0, 0], [+1, +1, +1]]))
    contour = np.abs(edge_x) + np.abs(edge_y)

    for _ in range(width - 1):
        contour = ndimage.convolve(contour, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))

    return np.int32(contour != 0)


def calculate_mask_weights(mask):
    salt_mean = 0.247966

    contour = calculate_contour(mask)

    weights = np.zeros_like(mask, np.float32)
    weights[mask == 0] = salt_mean / (1.0 - salt_mean)
    weights[mask == 1] = 1.0
    weights[contour == 1] = 3.0

    return weights


# https://www.microsoft.com/developerblog/2018/05/17/using-otsus-method-generate-data-training-deep-learning-image-segmentation-models/
def calculate_otsu_mask(image):
    image = 255 * image
    image = np.stack((image,) * 3, -1)
    image = image.astype(np.uint8)
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(image_grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] / 255


def crf_batch(images, masks):
    with Pool(16) as pool:
        return [c for c in pool.starmap(crf, zip(images, masks))]


def crf(image, mask):
    # Converting annotated image to RGB if it is Gray scale
    if len(mask.shape) < 3:
        mask = gray2rgb(mask)

    # Converting the annotations RGB color to single 32 bit integer
    annotated_label = mask[:, :, 0] + (mask[:, :, 1] << 8) + (mask[:, :, 2] << 16)

    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2

    # Setting up the CRF model
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 10 steps
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((image.shape[0], image.shape[1]))


# Source https://www.kaggle.com/bguberfain/unet-with-depth
def rlenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


def rldec(rle_mask):
    '''
    rle_mask: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    if rle_mask == "nan":
        return np.zeros((101, 101))
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101, 101).transpose((1, 0))


def postprocess_mask(mask):
    mask = remove_non_salt_inside_salt(mask)
    mask = remove_small_salt_areas(mask)
    return mask


def remove_non_salt_inside_salt(mask):
    mask = mask.copy()
    target = np.abs(1 - mask.astype(np.int8))
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(target, 4, cv2.CV_32S)
    for l in range(1, nlabels):
        stat = stats[l]
        left = stat[cv2.CC_STAT_LEFT]
        top = stat[cv2.CC_STAT_TOP]
        right = left + stat[cv2.CC_STAT_WIDTH]
        bottom = top + stat[cv2.CC_STAT_HEIGHT]
        if left > 0 and top > 0 and right < mask.shape[1] and bottom < mask.shape[0]:
            mask[labels == l] = 1.0
    return mask


def remove_small_salt_areas(mask):
    mask = mask.copy()
    target = mask.astype(np.uint8)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(target, 4, cv2.CV_32S)
    for l in range(1, nlabels):
        stat = stats[l]
        area = stat[cv2.CC_STAT_AREA]
        if area < 20:
            mask[labels == l] = 0.0
    return mask
