from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import tqdm
from skimage.feature import greycomatrix, greycoprops
from skimage.io import imread


def read_image(imgid):
    fn = '../salt/input/train/images/{}.png'.format(imgid)
    return imread(fn)[..., 0].astype(np.float32) / 255


def read_mask(imgid):
    fn = '../salt/input/train/masks/{}.png'.format(imgid)
    return imread(fn).astype(np.uint8)


def glcm_props(patch):
    lf = []
    # props = ['dissimilarity', 'contrast', 'homogeneity', 'energy', 'correlation']
    props = ['dissimilarity', 'contrast', 'homogeneity', 'energy']

    # left nearest neighbor
    glcm = greycomatrix(patch, [1], [0], 256, symmetric=True, normed=True)
    for f in props:
        lf.append(greycoprops(glcm, f)[0, 0])

    # upper nearest neighbor
    glcm = greycomatrix(patch, [1], [np.pi / 2], 256, symmetric=True, normed=True)
    for f in props:
        lf.append(greycoprops(glcm, f)[0, 0])

    # for i in range(len(props)):
    #     lf.append(0.5 * (lf[i] + lf[i + len(props)]))

    return lf


def patch_gen(img, PAD=4):
    img1 = (img * 255).astype(np.uint8)

    W = 101
    imgx = np.zeros((101 + PAD * 2, 101 + PAD * 2), dtype=img1.dtype)
    imgx[PAD:W + PAD, PAD:W + PAD] = img1
    imgx[:PAD, PAD:W + PAD] = img1[PAD:0:-1, :]
    imgx[-PAD:, PAD:W + PAD] = img1[W - 1:-PAD - 1:-1, :]
    imgx[:, :PAD] = imgx[:, PAD * 2:PAD:-1]
    imgx[:, -PAD:] = imgx[:, W + PAD - 1:-PAD * 2 - 1:-1]

    xx, yy = np.meshgrid(np.arange(0, W), np.arange(0, W))
    xx, yy = xx.flatten() + PAD, yy.flatten() + PAD

    for x, y in zip(xx, yy):
        patch = imgx[y - PAD:y + PAD + 1, x - PAD:x + PAD + 1]
        yield patch


def glcm_feature(img, verbose=False):
    W, NF, PAD = 101, 8, 4

    if img.sum() == 0:
        return np.zeros((W, W, NF), dtype=np.float32)

    # l = []
    # with Pool(3) as pool:
    #     for p in tqdm.tqdm(pool.imap(glcm_props, patch_gen(img, PAD)), total=W * W, disable=not verbose):
    #         l.append(p)
    l = [glcm_props(p) for p in patch_gen(img, PAD)]

    fimg = np.array(l, dtype=np.float32).reshape(101, 101, -1)
    return fimg


def calculate_glcm_features(img, verbose=False):
    fimg = glcm_feature(img, verbose)

    amin = np.amin(fimg, axis=(0, 1))
    amax = np.amax(fimg, axis=(0, 1))
    fimg = (fimg - amin) / (amax - amin)

    # fimg[..., 4] = np.power(fimg[..., 4], 3)
    # fimg[..., 9] = np.power(fimg[..., 9], 3)

    return fimg


def calculate_and_save_glcm_features(imgid, verbose=False):
    fimg = calculate_glcm_features(read_image(imgid), verbose)

    cv2.imwrite("../salt/input/glcm/dissimilarity-0/{}.png".format(imgid), (255 * fimg[..., 0]).astype(np.uint8))
    cv2.imwrite("../salt/input/glcm/dissimilarity-90/{}.png".format(imgid), (255 * fimg[..., 1]).astype(np.uint8))

    cv2.imwrite("../salt/input/glcm/contrast-0/{}.png".format(imgid), (255 * fimg[..., 2]).astype(np.uint8))
    cv2.imwrite("../salt/input/glcm/contrast-90/{}.png".format(imgid), (255 * fimg[..., 3]).astype(np.uint8))

    cv2.imwrite("../salt/input/glcm/homogeneity-0/{}.png".format(imgid), (255 * fimg[..., 4]).astype(np.uint8))
    cv2.imwrite("../salt/input/glcm/homogeneity-90/{}.png".format(imgid), (255 * fimg[..., 5]).astype(np.uint8))

    cv2.imwrite("../salt/input/glcm/energy-0/{}.png".format(imgid), (255 * fimg[..., 6]).astype(np.uint8))
    cv2.imwrite("../salt/input/glcm/energy-90/{}.png".format(imgid), (255 * fimg[..., 7]).astype(np.uint8))


def main():
    train_df = pd.read_csv("{}/train.csv".format("../salt/input"), index_col="id", usecols=[0])
    depths_df = pd.read_csv("{}/depths.csv".format("../salt/input"), index_col="id")
    train_df = train_df.join(depths_df)

    with Pool(4) as pool:
        for _ in tqdm.tqdm(pool.imap(calculate_and_save_glcm_features, train_df.index), total=len(train_df.index)):
            pass


if __name__ == "__main__":
    main()
