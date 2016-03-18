from skimage import data
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.feature_extraction import image
from sklearn.decomposition import PCA
import scipy.io as sio


class ANNField:

    def __init__(self, a, b, patch_size):
        self.img_A = a
        self.img_B = b
        # Assume patch_w = patch_h
        self.patch_width = patch_size
        self.patch_height = patch_size
        self.nearest_neighbors = 1
        self.ann_field = self.build_ann_field()

    def build_ann_field(self):
        # Create patches vectors from image vectors.
        print("Converting image A to patches...")
        patches_a = self.patchify(self.img_A, self.patch_height, self.patch_width)
        pca = PCA(n_components=10)
        print("Applying dimensionality reduction")
        # pca.fit(patchesA)
        # patchesA = pca.transform(patchesA)
        print("Image A converted.")
        print("Converting image B to patches...")
        patches_b = self.patchify(self.img_B, self.patch_height, self.patch_width)
        print("Applying dimensionality reduction")
        # pca.fit(patchesB)
        # patchesB = pca.transform(patchesB)
        print("Image B converted.")

        # Fit and find k-NN.
        print("Fitting k-NN...")
        nbrs = NearestNeighbors(n_neighbors=self.nearest_neighbors, algorithm='kd_tree').fit(patches_b)
        print("k-NN fitted.")
        print("Finding k-NN...")
        distances, indices = nbrs.kneighbors(patches_a)
        print("k-NN found.")

        # Build the NN-field from distances and indices
        print("Building NN-Field...")
        # Initialize empty nn-field of size imgB_height * imgB_width * 3
        self.ann_field = np.zeros((self.img_B.shape[0] - self.patch_height + 1,
                                   self.img_B.shape[1] - self.patch_width + 1, 3))
        # Create list of all coordinates in order to insert pixel patch indices into the right row and columns
        coordinates = [(y, x) for y in range(0, self.img_B.shape[0] - self.patch_height + 1)
                       for x in range(0, self.img_B.shape[1] - self.patch_width + 1)]
        for i in range(0, indices.shape[0]):
            # Map pixel indices to actual image coordinates
            y_a = coordinates[i][0]
            x_a = coordinates[i][1]
            y_b = coordinates[indices[i]][0]
            x_b = coordinates[indices[i]][1]
            # Place all x coordinates of the nearest neighbors into the first layer of the NN-Field.
            self.ann_field[y_a][x_a][0] = x_b
            # Place all y coordinates of the nearest neighbors into the second layer of the NN-Field.
            self.ann_field[y_a][x_a][1] = y_b

        # Finally, the third layer of the NN-field contains all the L2 distances
        # Value on nn_field[y][x][2] means the L2 dist to the nearest patch in B for the patch on position (y, x) in A
        dist = distances.reshape(self.img_B.shape[0] - self.patch_height + 1,
                                 self.img_B.shape[1] - self.patch_width + 1)
        self.ann_field[:, :, 2] = dist
        print("Finished.")
        return self.ann_field

    def write_mat(self):
        # Write to .mat file to be imported in MATLAB
        sio.savemat("..\\output\\ann_field.mat", {'ann_field': self.ann_field})

    def show_field(self):
        # Plot the ANN-field and original images
        _, ax = plt.subplots(ncols=2, nrows=2)
        ax[0][0].imshow(self.ann_field[:, :, 2], cmap='Greys_r')
        ax[0][1].imshow(self.ann_field[:, :, 1], cmap='Greys_r')
        ax[1][0].imshow(self.img_A)
        ax[1][1].imshow(self.img_B)
        plt.show()

    @staticmethod
    def patchify(img, patch_height, patch_width):
        img = image.extract_patches_2d(img, (patch_height, patch_width))
        (nr_of_patches, dimensions) = (img.shape[0], img.shape[1] * img.shape[2] * img.shape[3])
        return img.reshape(nr_of_patches, dimensions)  # Reshape to have a flattened representation of a pixel patch

A = data.imread("..\\img\\vidpair211.jpg")
B = data.imread("..\\img\\vidpair212.jpg")
field = ANNField(A, B, 3)
field.write_mat()
field.show_field()