import os
import sys
import imageio.v3 as iio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

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

def compare_all_method(l=[]):
    dataset_folder = "qsd1_w3/"
    if len(l)>0:
        list_path = l
    else:
        list_path = get_all_jpg_images(dataset_folder)
    # compare 3 methods
    for image_path in list_path:
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


def test_ssim_psnr(sigma_value=1.0, kernel_size=3, plot=True):
    queryset_folder = "qsd1_w3/"
    dataset_folder = "qsd1_w3/non_augmented/"
    query_set = get_all_jpg_images(queryset_folder)
    data_set = get_all_jpg_images(dataset_folder)
    gaussian_results = ([],[])
    median_results = ([],[])
    uniform_results = ([],[])

    for query_image_path, data_image_path in zip(query_set, data_set):
        query_image = iio.imread(query_image_path)
        gaussian_image = noise_remove(query_image, type_filter="Gaussian", sigma=sigma_value)
        median_image = noise_remove(query_image, type_filter="Median", kernel_size=kernel_size)
        uniform_image = noise_remove(query_image, type_filter="Uniform", kernel_size=kernel_size)
        real_image = iio.imread(data_image_path)
        
        gaussian_results[0].append(ssim(gaussian_image, real_image, channel_axis=2))
        gaussian_results[1].append(psnr(gaussian_image, real_image))
        median_results[0].append(ssim(median_image, real_image, channel_axis=2))
        median_results[1].append(psnr(median_image, real_image))
        uniform_results[0].append(ssim(uniform_image, real_image, channel_axis=2))
        uniform_results[1].append(psnr(uniform_image, real_image))
        difference_query_real = (ssim(query_image, real_image, channel_axis=2), psnr(query_image, real_image))
        #plot
        if plot:
            f, axarr = plt.subplots(3,2)
            f.suptitle(f"Test for all methods kernel {kernel_size:.2f} and sigma {sigma_value:.2f}f")
            axarr[0,0].imshow(query_image)
            axarr[0,0].get_xaxis().set_visible(False)
            axarr[0,0].get_yaxis().set_visible(False)
            axarr[0,0].set_title(f"Query image, ssim: {difference_query_real[0]:.2f} psnr:{difference_query_real[1]:.2f}")

            axarr[0,1].imshow(gaussian_image)
            axarr[0,1].get_xaxis().set_visible(False)
            axarr[0,1].get_yaxis().set_visible(False)
            axarr[0,1].set_title(f"Gaussian image, ssim: {gaussian_results[0][-1]:.2f} psnr:{gaussian_results[1][-1]:.2f}")

            axarr[1,0].imshow(median_image)
            axarr[1,0].get_xaxis().set_visible(False)
            axarr[1,0].get_yaxis().set_visible(False)
            axarr[1,0].set_title(f"Median image, ssim: {median_results[0][-1]:.2f} psnr:{median_results[1][-1]:.2f}")

            axarr[1,1].imshow(uniform_image)
            axarr[1,1].get_xaxis().set_visible(False)
            axarr[1,1].get_yaxis().set_visible(False)
            axarr[1,1].set_title(f"Uniform image, ssim: {uniform_results[0][-1]:.2f} psnr:{uniform_results[1][-1]:.2f}")

            axarr[2,0].imshow(real_image)
            axarr[2,0].get_xaxis().set_visible(False)
            axarr[2,0].get_yaxis().set_visible(False)
            axarr[2,0].set_title(f"Data image")
            
            axarr[2,1].get_xaxis().set_visible(False)
            axarr[2,1].get_yaxis().set_visible(False)
            plt.show()

        ssims = (sum(gaussian_results[0])/len(gaussian_results[0]),sum(median_results[0])/len(median_results[0]),sum(uniform_results[0])/len(uniform_results[0]))
        psnrs =  (sum(gaussian_results[1])/len(gaussian_results[1]),sum(median_results[1])/len(median_results[1]),sum(uniform_results[1])/len(uniform_results[1]))
    return (ssims, psnrs)
        


def test_all_ssim_psnr():
    test_sigma = [1.0,2.0,3.0,5.0,6.0,7.0,8.0]
    test_kernel = [3,5, 7, 11, 13, 17, 19, 23]
    g_result_ssim = [0,0,0,0,0,0,0,0]
    g_result_psnr = [0,0,0,0,0,0,0,0]
    m_result_ssim = [0,0,0,0,0,0,0,0]
    m_result_psnr = [0,0,0,0,0,0,0,0]
    u_result_ssim = [0,0,0,0,0,0,0,0]
    u_result_psnr = [0,0,0,0,0,0,0,0]
    for i,(kernel, sigma) in enumerate(zip(test_kernel, test_sigma)):
        ((ssim_gaussian, ssim_median, ssim_uniform),(psnr_gaussian, psnr_median, psnr_uniform)) = test_ssim_psnr(sigma_value=sigma, kernel_size=kernel,plot=False)
        print(f"Ssim gaussian: {ssim_gaussian},Ssim median: {ssim_median},Ssim uniform: {ssim_uniform}")
        print(f"psnr gaussian: {psnr_gaussian},psnr median: {psnr_median},psnr uniform: {psnr_uniform}")
        g_result_ssim[i] = ssim_gaussian
        g_result_psnr[i] = psnr_gaussian
        m_result_ssim[i] = ssim_median
        m_result_psnr[i] = psnr_median
        u_result_ssim[i] = ssim_uniform
        u_result_psnr[i] = psnr_uniform

    plt.plot(test_sigma, g_result_ssim)
    plt.plot(test_sigma, g_result_psnr)
    plt.show()

    plt.plot(test_kernel, m_result_ssim)
    plt.plot(test_kernel, m_result_psnr)
    plt.plot(test_kernel, u_result_ssim)
    plt.plot(test_kernel, u_result_psnr)
    plt.show()

if __name__ == '__main__':
    interesting_images = ["qsd1_w3/00000.jpg","qsd1_w3/00004.jpg","qsd1_w3/00016.jpg", "qsd1_w3/00006.jpg", "qsd1_w3/00017.jpg", "qsd1_w3/00018.jpg", "qsd1_w3/00027.jpg", "qsd1_w3/00028.jpg"] #textures, noise and some times even median is wrong
    output_folder = "denoised/"
    results_folder = "data/week2/results"

    compare_all_method(interesting_images)
    

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
    