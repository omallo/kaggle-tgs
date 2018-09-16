import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from skimage.color import gray2rgb

"""
Function which returns the labelled image after applying CRF
"""


# Original_image = Image which has to labelled
# Mask image = Which has been labelled by some technique..
def crf(original_image, mask_img):
    # Converting annotated image to RGB if it is Gray scale
    if len(mask_img.shape) < 3:
        mask_img = gray2rgb(mask_img)

    # Converting the annotations RGB color to single 32 bit integer
    annotated_label = mask_img[:, :, 0] + (mask_img[:, :, 1] << 8) + (mask_img[:, :, 2] << 16)

    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2

    # Setting up the CRF model
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 10 steps
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0], original_image.shape[1]))
