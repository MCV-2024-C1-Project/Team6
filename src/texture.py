import os
import imageio.v3 as iio
import skimage.color 
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Dictionary to hold different histogram components (if needed)
HistogramComponents = {}

# Factory for creating texture extractors
def TextureExtractorFactory(type:str, histogram_bins:int = 256):
    if type == "LBP":
        return LBPExtractor(histogram_bins)
    elif type == "DCT":
        return DCTExtractor(histogram_bins)
    elif type == "Wavelet":
        return WaveletExtractor(histogram_bins)
    else:
        sys.exit(f"ERROR: Unknown texture extraction type '{type}'")

# Base class for histogram extractors
class TextureExtractor(object):
    def __init__(self, histogram_bins:int = 256):
        self.histogram_bins = histogram_bins

    def extract(self, image, normalize=True):
        sys.exit("ERROR: The extract method should be implemented by a subclass")

#Values for R and P from presentation
#           1     8
#           2.5   12
#           4     16
class LBPExtractor(TextureExtractor):
    def __init__(self, histogram_bins:int = 256, R=1, P=8):
        super(LBPExtractor, self).__init__(histogram_bins)
        self.radius = R
        self.n_points = P

    def extract(self, image, normalize=True):
        if len(image.shape) == 3:
            gray_image = skimage.color.rgb2gray(image)
        else:
            gray_image = image

        lbp = local_binary_pattern(gray_image, self.P, self.R)
        print(lbp.shape)
        plt.imshow(lbp,cmap='gray')
        plt.show()
        hist, _ = np.histogram(lbp, bins=np.arange(0, self.n_points + 3), range=(0, self.n_points + 2), density=normalize)
        
        return hist

class DCTExtractor(TextureExtractor):
    def __init__(self, histogram_bins:int = 256):
        super(DCTExtractor, self).__init__(histogram_bins)

    def extract(self, image, normalize=True):
        return None

class WaveletExtractor(TextureExtractor):
    def __init__(self, histogram_bins:int = 256, wavelet='haar'):
        super(WaveletExtractor, self).__init__(histogram_bins)
        self.wavelet = wavelet

    def extract(self, image, normalize=True):
        return None

class BlockHistogramExtractor(TextureExtractor):
    def __init__(self, histogram_bins:int = 256, hist_type:str= "HSV", number_edge_block:int = 2):
        self.hist_type = hist_type
        self.number_edge_block = number_edge_block
        super(BlockHistogramExtractor, self).__init__(histogram_bins)
    
    def extract(self, image, normalize = True):

        sizei = image.shape[0]
        sizej = image.shape[1]

        sizei_block = int(sizei/self.number_edge_block)
        sizej_block = int(sizej/self.number_edge_block)

        # imgplot = plt.imshow(image)
        #plt.show()

        image_vectors = []
        for i in range(self.number_edge_block):
            for j in range(self.number_edge_block):
                i_left_bound = (sizei_block)*i
                i_right_bound = (sizei_block*(i+1))
                j_up_bound = (sizej_block)*j
                j_down_bound = (sizej_block*(j+1))
                # print("Square -----------------------")
                # print(f"i bound [{i_left_bound},{i_right_bound}]")
                # print(f"j bound[{j_up_bound},{j_down_bound}]")
                # print("-----------------------")
                block_image = image[i_left_bound:i_right_bound, j_up_bound:j_down_bound, :]
                # imgplot = plt.imshow(block_image)
                # plt.show()
                image_vectors.append(block_image)
        
        hist = TextureExtractorFactory(self.hist_type, self.histogram_bins)
        hist_vector = [] #each histogram is a vector of histograms so we flaten them 
        for subimage in image_vectors:
            hist_vector = hist_vector + hist.extract(subimage, normalize)

        return hist_vector

if __name__ == "__main__":

    image = iio.imread('target/BBDD/bbdd_00003.jpg')


    extractor = TextureExtractorFactory("LBP")
    
    # Extract features
    features = extractor.extract(image)
    print(image.shape)
    print(features)
