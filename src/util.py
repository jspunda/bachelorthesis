from sklearn.feature_extraction import image
import numpy as np
import math


def patchify(img, patch_height, patch_width):
    img = image.extract_patches_2d(img, (patch_height, patch_width))
    (nr_of_patches, dimensions) = (img.shape[0], img.shape[1] * img.shape[2] * img.shape[3])
    return np.reshape(img, (nr_of_patches, dimensions))  # Reshape to have a flattened representation of a pixel patch


def dist(a, b):
    return np.sqrt(np.sum((a-b)**2))
