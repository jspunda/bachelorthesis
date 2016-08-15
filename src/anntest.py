import annfield
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from skimage import data
import scipy.io as sio


class ANNTest:

    def __init__(self, pairs, patch_size, dim_red=-1, pca_fit=1, dataset="large"):
        self.pairs = pairs
        self.field = []
        self.patch_size = patch_size
        self.average_time = 0
        self.average_l2 = 0
        self.dim_red = dim_red
        self.dataset = dataset
        self.pca_fit = pca_fit
        if self.dim_red == -1:
            self.filename = str(self.pairs) + "Pairs" + str(patch_size) + "Patchsize" + "NoPCA.txt"
        else:
            self.filename = str(self.pairs) + "Pairs" + str(patch_size) + "Patchsize" + str(dim_red) + "PCA.txt"
        self.run_test()

    def run_test(self):
        total_time = 0
        total_l2 = 0
        for i in range(1, self.pairs*2, 2):
            print("Testing image pair", math.ceil(i/2))
            if self.dataset == "large":
                filename_a = "E:\\STACK\Bachelor Thesis\\Vidpairs_Dataset\\vidpair" + str(i) + ".jpg"
                filename_b = "E:\\STACK\Bachelor Thesis\\Vidpairs_Dataset\\vidpair" + str(i + 1) + ".jpg"
            else:
                if self.dataset == "small":
                    filename_a = "E:\\STACK\Bachelor Thesis\\Vidpairs_Dataset\\small\\vidpair" + str(i) + ".jpg"
                    filename_b = "E:\\STACK\Bachelor Thesis\\Vidpairs_Dataset\\small\\vidpair" + str(i + 1) + ".jpg"
            img_a = data.imread(filename_a)
            img_b = data.imread(filename_b)
            start = time.time()
            self.field = annfield.ANNField(img_a, img_b, self.patch_size, self.dim_red, self.pca_fit)
            total = time.time() - start
            total_time += total
            total_l2 += (np.mean(self.field.ann_field[:, :, 2]))
        self.average_time = total_time / self.pairs
        self.average_l2 = total_l2 / self.pairs

    def print_result(self):
        if self.dim_red > -1:
            print(self.pairs, "pairs,", "PCA to", self.dim_red, "dimensions, patch size", self.patch_size)
        else:
            print(self.pairs, "pairs, no PCA, patch size", self.patch_size)
        print("Average time:", self.average_time, "sec")
        print("Average L2 distance:", self.average_l2)

    def write_result(self):
        f = open("..\\output\\" + self.filename, 'w')
        f.write("Average time per pair: " + str(self.average_time) + " sec\n")
        f.write("Average L2 distance: " + str(self.average_l2) + '\n')

    def plot_result(self):
        plt.scatter(round(self.average_time, 2), round(self.average_l2, 2))
        plt.ticklabel_format(style="plain", useOffset=False)
        plt.show()

# How to run sample test
test = ANNTest(2, 3, 10, 0.1, "small")
test.print_result()

# Plot ANN field for last pair in test
test.field.show_field()

# Running a series of tests and gathering averages
# times = []
# averages = []

# for i in range(2, 10):
#     test = ANNTest(10, 3, i, 0.1, "small")
#     test.print_result()
#     times.append(test.average_time)
#     averages.append(test.average_l2)

# print(times, averages)

# Write to .mat file
#
# sio.savemat("..\\output\\test", {"times": times,
#                                 "averages": averages})

# How to plot the results
# plt.plot(times, averages, 'bo')
# plt.plot(times, averages)
# plt.xlabel('Seconds')
# plt.ylabel('L2 dist')
# title = str(pairs) + " pairs, patch size " + str(patchsize) + ", pca to " + str(pca)
# if imagesize == "small":
#     title += ", image size 500*208"
# else:
#     title += ", image size 1920*1080"
# plt.title(title)
# plt.ticklabel_format(style="plain", useOffset=False)
# plt.show()

