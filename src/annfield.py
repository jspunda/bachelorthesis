from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import random
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
        res_a = self.img_A
        res_b = self.img_B
        res_a = np.reshape(res_a, (res_a.shape[0]*res_a.shape[1], 3))
        res_b = np.reshape(res_a, (res_b.shape[0]*res_b.shape[1], 3))

        # Create patches vectors from image vectors.
        print("Converting images to patches...")
        pca = PCA(n_components=self.dim_red)
        patches_a = util.patchify(self.img_A, self.patch_height, self.patch_width)
        patches_b = util.patchify(self.img_B, self.patch_height, self.patch_width)
        patches_a_old = patches_a
        patches_b_old = patches_b
        print("Images converted.")
        if self.dim_red > -1:
            print("Applying dimensionality reduction")
            print("Creating samples")
            a = patches_a[np.random.choice(patches_a.shape[0], 15, replace=False)]
            b = patches_b[np.random.choice(patches_b.shape[0], 15, replace=False)]
            print("Done")
            pca.fit(np.concatenate((a, b), axis=0))
            print("PCA fitted")
            patches_a = pca.transform(patches_a)
            patches_b = pca.transform(patches_b)
            print("PCA transformed")

        # Fit and find k-NN.
        print("Fitting k-NN...")
        neighbors = NearestNeighbors(n_neighbors=self.nearest_neighbors, algorithm="kd_tree").fit(patches_b)
        print("k-NN fitted.")
        print("Finding k-NN...")
        distances, indices = neighbors.kneighbors(patches_a)
        print("k-NN found.")

        # Build the NN-field from distances and indices
        print("Building NN-Field...")
        # Initialize empty nn-field of size imgB_height * imgB_width * 3
        self.ann_field = np.zeros((self.img_B.shape[0] - self.patch_height + 1,
                                   self.img_B.shape[1] - self.patch_width + 1, 3))
        # Create list of all coordinates in order to insert pixel patch indices into the right row and columns
        coordinates = [(y, x) for y in range(0, self.img_B.shape[0] - self.patch_height + 1)
                       for x in range(0, self.img_B.shape[1] - self.patch_width + 1)]
        distances1 = []
        for i in range(0, indices.shape[0]):
            # Map pixel indices to actual image coordinates
            y_a = coordinates[i][0]
            x_a = coordinates[i][1]
            y_b = coordinates[indices[i]][0]
            x_b = coordinates[indices[i]][1]
            # Compute L2 dist in original space
            di = np.linalg.norm(np.array(patches_a_old[i], dtype=np.float64) -
                                np.array(patches_b_old[indices[i]][0], dtype=np.float64))
            distances1.append(di)
            # Place all x coordinates of the nearest neighbors into the first layer of the NN-Field.
            self.ann_field[y_a][x_a][0] = x_b
            # Place all y coordinates of the nearest neighbors into the second layer of the NN-Field.
            self.ann_field[y_a][x_a][1] = y_b
        # Finally, the third layer of the NN-field contains all the L2 distances
        # Value on nn_field[y][x][2] means the L2 dist to the nearest patch in B for the patch on position (y, x) in A
        dist = distances.reshape(self.img_B.shape[0] - self.patch_height + 1,
                                 self.img_B.shape[1] - self.patch_width + 1)
        distances1 = np.array(distances1)
        dist2 = distances1.reshape(self.img_B.shape[0] - self.patch_height + 1,
                                   self.img_B.shape[1] - self.patch_width + 1)
        print(np.mean(dist), np.mean(dist2))
        self.ann_field[:, :, 2] = dist2
        print("Finished.")
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
