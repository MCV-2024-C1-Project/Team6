import os
import imageio.v3 as iio
import skimage.color 
import numpy as np
import sys
import cv2

intermediate_image_path="intermediate_imgs/"
HistogramComponents = {
    'RGB': ['Red', 'Green', 'Blue'],
    'GRAY': ['Gray'],
    'HSV': ['Hue', 'Saturation', 'Value'],
    'YCbCr': ['Luma', 'Cb', 'Cr'],
    'Super': ['Luma', 'Cb', 'Cr']+['Hue', 'Saturation', 'Value'],
    
}
# 'Block of N for super': ['Luma0', 'Cb0', 'Cr0','Hue0', 'Saturation0', 'Value0', ... 'LumaN', 'CbN', 'CrN','HueN', 'SaturationN', 'ValueN']
# 'Piramid of N for super': ['Luma0', 'Cb0', 'Cr0','Hue0', 'Saturation0', 'Value0', ... 'LumaN', 'CbN', 'CrN','HueN', 'SaturationN', 'ValueN']
def HistogramExtractorFactory(type:str, histogram_bins:int = 256):
    if type == "RGB":
        return RGBHistogramExtractor(histogram_bins)
    elif type == "GRAY":
        return GrayHistogramExtractor(histogram_bins)
    elif type == "HSV":
        return HSVHistogramExtractor(histogram_bins)
    elif type == "YCbCr":
        return YCbCrHistogramExtractor(histogram_bins)
    elif type == 'Super':
        return SuperHistogram(histogram_bins)
    elif type[:4] == 'Block':
        #Block_4_HSV
        _, str_number_blocks, hist_type = type.split("_")
        blocks = int(str_number_blocks)
        return BlockHistogramExtractor(histogram_bins, hist_type=hist_type, number_edge_block = blocks)
    elif type[:6] == 'Pyramid':
        #Pyramid_3_HSV
        _, str_levels, hist_type = type.split("_")
        levels = int(str_levels)
        return PyramidHistogramExtractor(histogram_bins, hist_type=block_type, levels = levels)
    else:
        sys.exit("ERROR: Unknow histogram type: "+type)


class HistogramExtractor(object):
    def __init__(self, histogram_bins:int = 256):
        self.histogram_bins = histogram_bins

    def extract(self, image, normalize = True):
        sys.exit("ERROR: The extract method should be implementeds by a subclass")
    
class GrayHistogramExtractor(HistogramExtractor):
    def __init__(self, histogram_bins:int = 256):
        super(GrayHistogramExtractor, self).__init__(histogram_bins)

    def extract(self, image, normalize = True):
        # Caution: image from imageio is RGB, from cv2 is BGR
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

    def extract(self, image, normalize = True):
        # Caution: image from imageio is RGB, from cv2 is BGR
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

    def extract(self, image_input, normalize = True):
        # Caution: image from imageio is RGB, from cv2 is BGR
        image = cv2.cvtColor(image_input, cv2.COLOR_RGB2HSV)
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

    def extract(self, image_input, normalize = True):
        # Caution: image from imageio is RGB, from cv2 is BGR
        image = cv2.cvtColor(image_input, cv2.COLOR_RGB2YCrCb)
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
    def extract(self, image, normalize = True):
        return self.e1.extract(image, normalize) + self.e2.extract(image, normalize)

class BlockHistogramExtractor(HistogramExtractor):
    def __init__(self, histogram_bins:int = 256, hist_type:str= "HSV", number_edge_block:int = 2):
        self.hist_type = hist_type
        self.number_edge_block = number_edge_block
        super(BlockHistogramExtractor, self).__init__(histogram_bins)
    
    def extract(self, image, normalize = True):

        sizei = im.shape[0]
        sizej = im.shape[1]

        sizei_block = sizei/number_edge_block
        sizej_block = sizej/number_edge_block

        image_vectors = []
        for i in range(number_edge_block):
            for j in range(number_edge_block):
                i_left_bound = (sizei_block)*i
                i_right_bound = (sizei_block*(i+1))-1
                j_up_bound = (sizej_block)*j
                j_down_bound = (sizej_block*(j+1))-1
                # print("Square -----------------------")
                # print(f"i bound [{i_left_bound},{i_right_bound}]")
                # print(f"j bound[{j_up_bound},{j_down_bound}]")
                # print("-----------------------")
                block_image = image[i_left_bound:i_right_bound, j_up_bound:j_down_bound, :]
                image_vectors.append(block_image)
        
        hist = HistogramExtractorFactory(self.hist_type, self.bin)
        hist_vector = [hist for hist in hist.extract(subimage, normalize) for subimage in image_vectors] #each histogram is a vector of histograms sao we flaten them 

        return hist_vector

class PyramidHistogramExtractor(HistogramExtractor):
    def __init__(self, histogram_bins:int = 256, hist_type:str= "HSV", levels:int = 2):
        self.hist_type = hist_type
        self.levels = levels
        super(BlockHistogramExtractor, self).__init__(histogram_bins)
    
    def extract(self, image, normalize = True):
        hist_level_one = HistogramExtractorFactory(self.hist_type, histogram_bins = self.histogram_bins) # level one is global histogram
        hist_vector = hist_level_one.extract(image, normalize)

        for i in range(1, levels):
            hist_level_n = HistogramExtractorFactory("Block", histogram_bins = self.histogram_bins, block_type = self.hist_type, blocks = pow(2,i))# level 2 2 blocks, level 3 4 blocks, level 4 8 blocks.... level N 2^(N-1)
            hist_vector = hist_vector + hist_level_n.extract(image, normalize)

        return hist_vector