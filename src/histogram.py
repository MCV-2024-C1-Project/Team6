import os
import imageio.v3 as iio
import skimage.color 
import numpy as np
import sys
import cv2

HistogramComponents = {
    'RGB': ['Red', 'Green', 'Blue'],
    'GRAY': ['Gray'],
    'HSV': ['Hue', 'Saturation', 'Value'],
    'YCbCr': ['Luma', 'Cb', 'Cr'],
    'Super': ['Luma', 'Cb', 'Cr']+['Hue', 'Saturation', 'Value'],
}

def HistogramExtractorFactory(type:str, histogram_bins:int = 256):
    if type == "RGB":
        return RGBHistogramExtractor(histogram_bins)
    elif type == "GRAY":
        return GrayHistogramExtractor(histogram_bins)
    elif type == "HSV":
        return HSVHistogramExtractor(histogram_bins)
    elif type == "YCbCr":
        return YCbCrHistogramExtractor(histogram_bins)
    elif type=='Super':
        return SuperHistogram(histogram_bins)
    else:
        sys.exit("ERROR: Unknow histogram type: "+type)

class HistogramExtractor(object):
    def __init__(self, histogram_bins:int = 256):
        self.histogram_bins = histogram_bins

    def extract(self, image_path:str, normalize = True):
        sys.exit("ERROR: The extract method should be implementeds by a subclass")
    
class GrayHistogramExtractor(HistogramExtractor):
    def __init__(self, histogram_bins:int = 256):
        super(GrayHistogramExtractor, self).__init__(histogram_bins)

    def extract(self, image_path:str, normalize = True):
        # Caution: image from imageio is RGB, from cv2 is BGR
        image = iio.imread(image_path)
        histograms = []

        #GRAY mode
        bin_edges = np.linspace(0, 1, num=self.histogram_bins + 1)
        gray_image = skimage.color.rgb2gray(image)
        histogram, bin_edges = np.histogram(
            gray_image.flatten(),
            bins=bin_edges
            )
        histogram = histogram / histogram.sum() if normalize else histogram
        histograms.append(histogram)

        return histograms

class RGBHistogramExtractor(HistogramExtractor):
    def __init__(self, histogram_bins:int = 256):
        super(RGBHistogramExtractor, self).__init__(histogram_bins)

    def extract(self, image_path:str, normalize = True):
        # Caution: image from imageio is RGB, from cv2 is BGR
        image = iio.imread(image_path)
        histograms = [] 
        # RGB mode
        bin_edges = np.linspace(0, 255, num=self.histogram_bins + 1)
        for channel in range(image.shape[2]):
            single_channel_img = image[:,:,channel]
            channel_histogram, bin_edges = np.histogram(
                single_channel_img.flatten(),
                bins=bin_edges
                )
            channel_histogram = channel_histogram / channel_histogram.sum() if normalize else channel_histogram
            histograms.append(channel_histogram)
        
        return histograms
    

class HSVHistogramExtractor(HistogramExtractor):
    def __init__(self, histogram_bins:int = 256):
        super(HSVHistogramExtractor, self).__init__(histogram_bins)

    def extract(self, image_path:str, normalize = True):
        # Caution: image from imageio is RGB, from cv2 is BGR
        image = iio.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        histograms = [] 
        # RGB mode
        bin_edges = np.linspace(0, 255, num=self.histogram_bins + 1)
        v_channel = image[:, :, -1]
        v_channel = ((v_channel - v_channel.mean()) / (v_channel.std()+1e-10))
        
        v_bin_edges = np.linspace(-10, 10, num=self.histogram_bins + 1)
        for channel in range(image.shape[2]-1):
            single_channel_img = image[:,:,channel]
            channel_histogram, bin_edges = np.histogram(
                single_channel_img.flatten(),
                bins=bin_edges
                )
            channel_histogram = channel_histogram / channel_histogram.sum() if normalize else channel_histogram
            histograms.append(channel_histogram)
        v_histogram, bin_edges = np.histogram(
                v_channel.flatten(),
                bins=v_bin_edges
                )
        v_histogram = v_histogram / v_histogram.sum() if normalize else v_histogram
        histograms.append(v_histogram)


        return histograms
    

class YCbCrHistogramExtractor(HistogramExtractor):
    def __init__(self, histogram_bins:int = 256):
        super(YCbCrHistogramExtractor, self).__init__(histogram_bins)

    def extract(self, image_path:str, normalize = True):
        # Caution: image from imageio is RGB, from cv2 is BGR
        image = iio.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        histograms = [] 
        # RGB mode
        bin_edges = np.linspace(0, 255, num=self.histogram_bins + 1)
        y_channel = image[:, :, 0]
        y_channel = ((y_channel - y_channel.mean()) / (y_channel.std()+1e-10))
        
        y_bin_edges = np.linspace(-10, 10, num=self.histogram_bins + 1)
        y_histogram, y_bin_edges = np.histogram(
                y_channel.flatten(),
                bins=y_bin_edges
                )
        y_histogram = y_histogram / y_histogram.sum() if normalize else y_histogram
        histograms.append(y_histogram)
        for channel in range(1,image.shape[2]):
            single_channel_img = image[:,:,channel]
            channel_histogram, bin_edges = np.histogram(
                single_channel_img.flatten(),
                bins=bin_edges
                )
            channel_histogram = channel_histogram / channel_histogram.sum() if normalize else channel_histogram
            histograms.append(channel_histogram)
        
        return histograms
    
class SuperHistogram(HistogramExtractor):
    def __init__(self, histogram_bins:int = 256):
        super(SuperHistogram, self).__init__(histogram_bins)
        self.e1 = YCbCrHistogramExtractor(histogram_bins)
        self.e2 = HSVHistogramExtractor(histogram_bins)
    def extract(self, image_path:str, normalize = True):
        return self.e1.extract(image_path, normalize) + self.e2.extract(image_path, normalize)