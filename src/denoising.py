import os
import sys
import imageio.v3 as iio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter

class Filter(object):
    def __init__(self, kernel:int = 3):
        self.kernel = kernel

    def apply(self, image):
        sys.exit("ERROR: The apply method should be implementeds by a subclass")
#low pass filter
class UniformFilter(Filter):
    def __init__(self, kernel:int = 3):
        super(UniformFilter, self).__init__(kernel)

    def apply(self, image):
        return uniform_filter(image, size=self.kernel)

class GaussianFilter(Filter):
    def __init__(self, kernel:int = 3, sigma = 1.0):
        super(GaussianFilter, self).__init__(kernel)
        self.sigma = sigma

    def apply(self, image):
        #print(f"kernel {self.kernel}, sigma {self.sigma}")
        return gaussian_filter(image, sigma=self.sigma)

class MedianFilter(Filter):
    def __init__(self, kernel:int = 3):
        super(MedianFilter, self).__init__(kernel)

    def apply(self, image):
        return median_filter(image, size=self.kernel)

# def denoise_reedge_filter(image, kernel_size=3, lowpass="Gaussian", highpass="Laplacian"):
#     pass
#high pass filter
class LaplacianFilter(Filter):
    def __init__(self, kernel:int = 3):
        super(MedianFilter, self).__init__(kernel)

    def apply(self, image):
        pass

def get_all_jpg_images(input_folder):
    image_paths = []
    for file in  os.listdir(input_folder):
        if file.endswith(".jpg"):
            path = input_folder+file
            image_paths.append(path)

    return sorted(image_paths)

def filter_selector(type_filter, kernel_size=3, sigma=1.0):
    if type_filter == "Gaussian":
        return GaussianFilter(sigma=sigma)
    elif type_filter == "Median":
        return MedianFilter(kernel_size)
    elif type_filter == "Uniform":
        return UniformFilter(kernel_size)
    else:
        sys.exit(f"ERROR: Unknow filter type: {type_filter}")

def noise_remove(image, type_filter="Gaussian", kernel_size=3, sigma=1.0):
    noise_deleter = filter_selector(type_filter, kernel_size, sigma)
    return noise_deleter.apply(image)

def noise_removal(image, filter_arguments):
    type_filter, args = filter_arguments.split("-")
    if type_filter == "Gaussian":
        return noise_remove(image, type_filter=type_filter, sigma=float(args))
    else:
        return noise_remove(image, type_filter=type_filter, kernel_size=int(args))

def test_kernel(type_filter):
    dataset_folder = "qsd1_w3/"
    test_kernel = [3,5,13,29,53]
    for image_path in get_all_jpg_images(dataset_folder):
        raw_image = iio.imread(image_path)
        list_kernel_image_test = [noise_remove(raw_image, type_filter=type_filter, kernel_size=kernel_size) for kernel_size in test_kernel]
        f, axarr = plt.subplots(len(list_kernel_image_test),1)
        f.suptitle(f"Teste {type_filter} value")
        for i, image in enumerate(list_kernel_image_test):
            axarr[i].imshow(image)
            axarr[i].get_xaxis().set_visible(False)
            axarr[i].get_yaxis().set_visible(False)
            axarr[i].set_title(f"Test kernel {str(test_kernel[i])}")
        plt.show()

def test_sigma():
    dataset_folder = "qsd1_w3/"
    test_sigma = [1.0,2.0,3.0,5.0,10.0,20.0]
    for image_path in get_all_jpg_images(dataset_folder):
        raw_image = iio.imread(image_path)
        list_sigma_image_test = [noise_remove(raw_image, type_filter="Gaussian", sigma=sigma) for sigma in test_sigma]
        f, axarr = plt.subplots(len(list_sigma_image_test),1)
        f.suptitle("Teste Gaussian value")
        for i, image in enumerate(list_sigma_image_test):
            axarr[i].imshow(image)
            axarr[i].get_xaxis().set_visible(False)
            axarr[i].get_yaxis().set_visible(False)
            axarr[i].set_title(f"Test sigma {str(test_sigma[i])}")
        plt.show()

def compare_all_method():
    dataset_folder = "qsd1_w3/"
    # compare 3 methods
    for image_path in get_all_jpg_images(dataset_folder):
        print(f"Image: {image_path}")
        raw_image = iio.imread(image_path)
        gaussian_image = noise_remove(raw_image, type_filter="Gaussian", sigma=1.0)
        median_image = noise_remove(raw_image, type_filter="Median", kernel_size=3)
        uniform_image = noise_remove(raw_image, type_filter="Uniform", kernel_size=3)
        #subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(2,2)
        f.suptitle(f"Test for all methods kernel 3 and sigma 1.0f")
        axarr[0,0].imshow(raw_image)
        axarr[0,0].get_xaxis().set_visible(False)
        axarr[0,0].get_yaxis().set_visible(False)
        axarr[0,0].set_title("Raw image")

        axarr[0,1].imshow(gaussian_image)
        axarr[0,1].get_xaxis().set_visible(False)
        axarr[0,1].get_yaxis().set_visible(False)
        axarr[0,1].set_title("Gaussian image")

        axarr[1,0].imshow(median_image)
        axarr[1,0].get_xaxis().set_visible(False)
        axarr[1,0].get_yaxis().set_visible(False)
        axarr[1,0].set_title("Median image")

        axarr[1,1].imshow(uniform_image)
        axarr[1,1].get_xaxis().set_visible(False)
        axarr[1,1].get_yaxis().set_visible(False)
        axarr[1,1].set_title("Uniform image")
        #imgplot = plt.imshow(raw_image)
        plt.show()

if __name__ == '__main__':
    interesting_images = ["qsd1_w3/00004.jpg","qsd1_w3/00016.jpg", "qsd1_w3/00006.jpg", "qsd1_w3/00017.jpg", "qsd1_w3/00018.jpg", "qsd1_w3/00027.jpg", "qsd1_w3/00028.jpg"] #textures, noise and some times even median is wrong
    output_folder = "denoised/"
    results_folder = "data/week2/results"
    compare_all_method()
    #test_kernel("Uniform")
    #test_sigma()
    # Get the list of all files and directories
    # raw_image = iio.imread("qsd1_w3/00006.jpg")
    # list_kernel_image_test = [noise_remove(raw_image, type_filter="Median", kernel_size=kernel_size) for kernel_size in test_kernel]
    # f, axarr = plt.subplots(len(list_kernel_image_test),1)
    # f.suptitle("Teste Median value")
    # for i, image in enumerate(list_kernel_image_test):
    #     axarr[i].imshow(image)
    #     axarr[i].get_xaxis().set_visible(False)
    #     axarr[i].get_yaxis().set_visible(False)
    #     axarr[i].set_title(f"Test kernel {str(test_kernel[i])}")
    # plt.show()
    