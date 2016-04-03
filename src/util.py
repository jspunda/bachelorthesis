from sklearn.feature_extraction import image
from sklearn.decomposition import PCA


def apply_pca(vector, components):
    pca = PCA(n_components=components)
    pca.fit(vector)
    return pca.transform(vector)


def patchify(img, patch_height, patch_width):
    img = image.extract_patches_2d(img, (patch_height, patch_width))
    (nr_of_patches, dimensions) = (img.shape[0], img.shape[1] * img.shape[2] * img.shape[3])
    return img.reshape(nr_of_patches, dimensions)  # Reshape to have a flattened representation of a pixel patch
