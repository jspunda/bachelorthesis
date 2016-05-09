from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import math
import util


class ANNField:

    def __init__(self, a, b, patch_size, dim_red=-1):
        self.img_A = a
        self.img_B = b
        # Assume patch_w = patch_h
        self.patch_width = patch_size
        self.patch_height = patch_size
        self.dim_red = dim_red
        self.nearest_neighbors = 1
        self.ann_field = self.build_ann_field

    @property
    def build_ann_field(self):
        # Create patches vectors from image vectors.
        # print("Converting images to patches...")
        pca = PCA(n_components=self.dim_red)
        patches_a = util.patchify(self.img_A, self.patch_height, self.patch_width)
        patches_b = util.patchify(self.img_B, self.patch_height, self.patch_width)
        patches_a_old = patches_a
        patches_b_old = patches_b
        # print("Images converted.")
        if self.dim_red > -1:
            # print("Applying dimensionality reduction")
            # print("Creating samples")
            a = patches_a[np.random.choice(patches_a.shape[0], 10, replace=False), :]
            b = patches_b[np.random.choice(patches_b.shape[0], 10, replace=False), :]
            # a = [i for i in range(10)]
            # b = [i for i in range(10)]
            # for i in range(0, 10):
            #     a[i] = np.random.randint(3, size=27)
            #     b[i] = np.random.randint(255, size=27)
            # print("Done")

            print(np.mean(a))
            print("sdfa")
            print(np.mean(b))
            pca.fit(b)
            # print("PCA fitted")
            patches_a = pca.transform(patches_a)
            patches_b = pca.transform(patches_b)
            print(np.sum(patches_a))
            print(np.sum(patches_b))
            # print("PCA transformed")

        # Fit and find k-NN.
        # print("Fitting k-NN...")
        neighbors = NearestNeighbors(n_neighbors=self.nearest_neighbors, algorithm="kd_tree").fit(patches_b)
        # print("k-NN fitted.")
        # print("Finding k-NN...")
        distances, indices = neighbors.kneighbors(patches_a)
        # print("k-NN found.")
        indices = indices.reshape(len(indices))

        # Build the NN-field from distances and indices
        # print("Building NN-Field...")
        # Initialize empty nn-field of size imgB_height * imgB_width * 3
        self.ann_field = np.zeros((self.img_B.shape[0] - self.patch_height + 1,
                                   self.img_B.shape[1] - self.patch_width + 1, 3))
        # Compute distances in original dimensional space (before PCA)
        # print("nn distances")
        distances_original = np.linalg.norm(np.array(patches_a_old, dtype=np.int32) -
                                            np.array(patches_b_old[indices, :], dtype=np.int32), axis=1)

        # Reshape indices to match original image width and height
        dist = distances_original.reshape(self.img_B.shape[0] - self.patch_height + 1,
                                          self.img_B.shape[1] - self.patch_width + 1)
        # Reshape indices to match original image width and height
        indices = indices.reshape(self.img_B.shape[0] - self.patch_height + 1,
                                  self.img_B.shape[1] - self.patch_width + 1)
        # Insert x coords into the first layer of the ann field
        self.ann_field[:, :, 0] = np.remainder(indices, (self.img_B.shape[1] - self.patch_height + 1))
        # Insert x coords into the second layer of the ann field
        self.ann_field[:, :, 1] = np.floor_divide(indices, (self.img_B.shape[1] - self.patch_width + 1))
        # The third layer of the NN-field contains all the L2 distances
        # Value on nn_field[y][x][2] means the L2 dist to the nearest patch in B for the patch on position (y, x) in A
        self.ann_field[:, :, 2] = dist
        # print("Finished.")
        return self.ann_field

    def write_mat(self):
        # Write to .mat file to be imported in MATLAB
        print("Writing ann field to .mat file...")
        if self.dim_red > -1:
            filename = "..\\output\\" + str(self.patch_width) + "Patchsize" + str(self.dim_red) + "PCA.mat"
        else:
            filename = "..\\output\\" + str(self.patch_width) + "PatchsizeNoPCA.mat"
        sio.savemat(filename, {"ann_field": self.ann_field})
        print("Done writing.")

    def show_field(self):
        # Plot the ANN-field and original images
        _, ax = plt.subplots(ncols=2, nrows=2)
        ax[0][0].set_title("L2 Distances")
        ax[0][1].set_title("Ann-Field X-Coords")
        ax[1][0].set_title("Original image A")
        ax[1][1].set_title("Original image B")
        ax[0][0].imshow(self.ann_field[:, :, 2])
        ax[0][1].imshow(self.ann_field[:, :, 1], cmap="Greys_r")
        ax[1][0].imshow(self.img_A)
        ax[1][1].imshow(self.img_B)
        plt.show()
