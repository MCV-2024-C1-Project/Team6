import os
import imageio.v3 as iio
import skimage.color 
import numpy as np
import sys

def HistogramExtractorFactory(type:str, histogram_bins:int = 256):
    if type == "RGB":
        return GrayHistogramExtractor(histogram_bins)
    elif type == "GRAY":
        return GrayHistogramExtractor(histogram_bins)
    else:
        sys.exit("ERROR: Unknow histogram type: "+type)

class HistogramExtractor(object):
    def __init__(self, histogram_bins:int = 256):
        self.histogram_bins = histogram_bins

    def extract(self, image_path:str, normalize = True):
        sys.exit("ERROR: The extract method should be implementeds by a subclass")
        # # Caution: image from imageio is RGB, from cv2 is RBG
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
        # Caution: image from imageio is RGB, from cv2 is RBG
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
        super(GrayHistogramExtractor, self).__init__(histogram_bins)

    def extract(self, image_path:str, normalize = True):
        # Caution: image from imageio is RGB, from cv2 is RBG
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
    

