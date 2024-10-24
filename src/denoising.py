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
        return gaussian_filter(image, sigma=self.sigma, radius=self.kernel)

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
        return GaussianFilter(kernel_size, sigma)
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
        kernel_size,sigma = args.split("_")
        return noise_remove(image, type_filter=type_filter, kernel_size=int(kernel_size), sigma=float(sigma))
    else:
        return noise_remove(image, type_filter=type_filter, kernel_size=int(args))


if __name__ == '__main__':
    dataset_folder = "qsd1_w3/"
    output_folder = "denoised/"
    results_folder = "data/week2/results"
    # Get the list of all files and directories
    for image_path in get_all_jpg_images(dataset_folder):
        print(f"Image: {image_path}")
        raw_image = iio.imread(image_path)
        gaussian_image = noise_remove(raw_image, type_filter="Gaussian")
        median_image = noise_remove(raw_image, type_filter="Median")
        uniform_image = noise_remove(raw_image, type_filter="Uniform")
        #subplot(r,c) provide the no. of rows and columns
        f, axarr = plt.subplots(2,2)
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