import annfield
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from skimage import data
import scipy.io as sio


class ANNTest:

    def __init__(self, pairs, patch_size, dim_red=-1, dataset="large"):
        self.pairs = pairs
        self.field = []
        self.patch_size = patch_size
        self.average_time = 0
        self.average_l2 = 0
        self.dim_red = dim_red
        self.dataset = dataset
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
            self.field = annfield.ANNField(img_a, img_b, self.patch_size, self.dim_red)
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

times = []
averages = []

test = ANNTest(1, 3, 10, "small")
test.print_result()
test.field.write_mat()
times.append(test.average_time)
averages.append(test.average_l2)

# test2 = ANNTest(10, 3, 2, "small")
# times.append(test2.average_time)
# averages.append(test2.average_l2)
#
# test3 = ANNTest(10, 3, 3, "small")
# times.append(test3.average_time)
# averages.append(test3.average_l2)
#
# test4 = ANNTest(10, 3, 4, "small")
# times.append(test4.average_time)
# averages.append(test4.average_l2)

# test5 = ANNTest(10, 2, 5, "small")
# times.append(test5.average_time)
# averages.append(test5.average_l2)

# sio.savemat("..\\output\\test", {"times": times,
#                                 "averages": averages})
#
# print(times, averages)
# plt.plot(times, averages, 'bo')
# plt.plot(times, averages)
# plt.ticklabel_format(style="plain", useOffset=False)
# plt.show()