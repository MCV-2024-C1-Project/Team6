import os
import imageio.v3 as iio
import skimage.color 
import numpy as np
import sys
import cv2

def HistogramExtractorFactory(type:str, histogram_bins:int = 256):
    if type == "RGB":
        return RGBHistogramExtractor(histogram_bins)
    elif type == "GRAY":
        return GrayHistogramExtractor(histogram_bins)
    elif type == "HSV":
        return HSVHistogramExtractor(histogram_bins)
    elif type == "YCbCr":
        return YCbCrHistogramExtractor(histogram_bins)
    else:
        sys.exit("ERROR: Unknow histogram type: "+type)

class HistogramExtractor(object):
    def __init__(self, histogram_bins:int = 256):
        self.histogram_bins = histogram_bins

    def extract(self, image_path:str, normalize = True):
        sys.exit("ERROR: The extract method should be implementeds by a subclass")
        # # Caution: image from imageio is RGB, from cv2 is BGR
        # image = iio.imread(image_path)
        # self.histograms = [] 
        # if self.color_mode == 'RGB':
        #     # RGB mode
        #     bin_edges = np.linspace(0, 255, num=self.histogram_bins + 1)

        #     for channel in range(image.shape[2]):
        #         single_channel_img = image[:,:,channel]
        #         channel_histogram, bin_edges = np.histogram(
        #             single_channel_img.flatten(),
        #             bins=bin_edges
        #             )
        #         channel_histogram = channel_histogram / channel_histogram.sum() if normalize else channel_histogram
        #         self.histograms.append(channel_histogram)

        # elif self.color_mode == 'GRAY':
        #     #GRAY mode
        #     bin_edges = np.linspace(0, 1, num=self.histogram_bins + 1)
        #     gray_image = skimage.color.rgb2gray(image)
        #     histogram, bin_edges = np.histogram(
        #         gray_image.flatten(),
        #         bins=bin_edges
        #         )
        #     histogram = histogram / histogram.sum() if normalize else histogram
        #     self.histograms.append(histogram)

        # else: # Placeholder
        #     pass
        
        # return self.histograms
    
class GrayHistogramExtractor(HistogramExtractor):
    def __init__(self, histogram_bins:int = 256):
        super(GrayHistogramExtractor, self).__init__(histogram_bins)

    def extract(self, image_path:str, normalize = True):
        # Caution: image from imageio is RGB, from cv2 is BGR
        image = iio.imread(image_path)
        self.histograms = []

        #GRAY mode
        bin_edges = np.linspace(0, 1, num=self.histogram_bins + 1)
        gray_image = skimage.color.rgb2gray(image)
        histogram, bin_edges = np.histogram(
            gray_image.flatten(),
            bins=bin_edges
            )
        histogram = histogram / histogram.sum() if normalize else histogram
        self.histograms.append(histogram)

        return self.histograms

class RGBHistogramExtractor(HistogramExtractor):
    def __init__(self, histogram_bins:int = 256):
        super(RGBHistogramExtractor, self).__init__(histogram_bins)

    def extract(self, image_path:str, normalize = True):
        # Caution: image from imageio is RGB, from cv2 is BGR
        image = iio.imread(image_path)
        self.histograms = [] 
        # RGB mode
        bin_edges = np.linspace(0, 255, num=self.histogram_bins + 1)
        for channel in range(image.shape[2]):
            single_channel_img = image[:,:,channel]
            channel_histogram, bin_edges = np.histogram(
                single_channel_img.flatten(),
                bins=bin_edges
                )
            channel_histogram = channel_histogram / channel_histogram.sum() if normalize else channel_histogram
            self.histograms.append(channel_histogram)
        
        return self.histograms
    

class HSVHistogramExtractor(HistogramExtractor):
    def __init__(self, histogram_bins:int = 256):
        super(HSVHistogramExtractor, self).__init__(histogram_bins)

    def extract(self, image_path:str, normalize = True):
        # Caution: image from imageio is RGB, from cv2 is BGR
        image = iio.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        self.histograms = [] 
        # RGB mode
        bin_edges = np.linspace(0, 255, num=self.histogram_bins + 1)
        for channel in range(image.shape[2]):
            single_channel_img = image[:,:,channel]
            channel_histogram, bin_edges = np.histogram(
                single_channel_img.flatten(),
                bins=bin_edges
                )
            channel_histogram = channel_histogram / channel_histogram.sum() if normalize else channel_histogram
            self.histograms.append(channel_histogram)


        return self.histograms
    

class YCbCrHistogramExtractor(HistogramExtractor):
    def __init__(self, histogram_bins:int = 256):
        super(YCbCrHistogramExtractor, self).__init__(histogram_bins)

    def extract(self, image_path:str, normalize = True):
        # # Caution: image from imageio is RGB, from cv2 is BGR
        # image = iio.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV )
        # self.histograms = [] 
        # # RGB mode
        # bin_edges = np.linspace(0, 255, num=self.histogram_bins + 1)
        # for channel in range(image.shape[2]):
        #     single_channel_img = image[:,:,channel]
        #     channel_histogram, bin_edges = np.histogram(
        #         single_channel_img.flatten(),
        #         bins=bin_edges
        #         )
        #     channel_histogram = channel_histogram / channel_histogram.sum() if normalize else channel_histogram
        #     self.histograms.append(channel_histogram)
        
        # return self.histograms
        return NotImplemented