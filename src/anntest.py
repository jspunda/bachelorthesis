import annfield
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from skimage import data


class ANNTest:

    def __init__(self, pairs, patch_size, dim_rec=-1):
        self.pairs = pairs
        self.patch_size = patch_size
        self.average_time = 0
        self.average_l2 = 0
        self.dim_rec = dim_rec
        if self.dim_rec == -1:
            self.filename = str(self.pairs) + "pairs" + str(patch_size) + "Patchsize" + "NoPCA.txt"
        else:
            self.filename = str(self.pairs) + "Pairs" + str(patch_size) + "Patchsize" + str(dim_rec) + "PCA.txt"
        self.run_test()

    def run_test(self):
        total_time = 0
        total_l2 = 0
        for i in range(1, self.pairs*2, 2):
            print("Testing image pair", math.ceil(i/2))
            filename_a = "E:\\STACK\Bachelor Thesis\\Vidpairs_Dataset\\small\\vidpair" + str(i) + ".jpg"
            filename_b = "E:\\STACK\Bachelor Thesis\\Vidpairs_Dataset\\small\\vidpair" + str(i + 1) + ".jpg"
            img_a = data.imread(filename_a)
            img_b = data.imread(filename_b)
            start = time.time()
            field = annfield.ANNField(img_a, img_b, self.patch_size, self.dim_rec)
            total = time.time() - start
            total_time += total
            total_l2 += (np.mean(field.ann_field[:, :, 2]))
        self.average_time = total_time / self.pairs
        self.average_l2 = total_l2 / self.pairs

    def print_result(self):
        if self.dim_rec > -1:
            print(self.pairs, "pairs,", "PCA to", self.dim_rec, "dimensions, patch size", self.patch_size)
        else:
            print(self.pairs, "pairs, no PCA, patch size", self.patch_size)
        print("Average time:", self.average_time, "sec")
        print("Average L2 distance:", self.average_l2)

    def write_test(self):
        f = open("..\\output\\" + self.filename, 'w')
        f.write("Average time per pair: " + str(self.average_time) + " sec\n")
        f.write("Average L2 distance: " + str(self.average_l2) + '\n')

    def plot_test(self):
        plt.scatter(round(self.average_time, 2), round(self.average_l2, 2))
        plt.ticklabel_format(style="plain", useOffset=False)
        plt.show()

test = ANNTest(10, 3)
test.print_result()
test.write_test()
# test.plot_test()