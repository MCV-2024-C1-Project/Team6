import os
import imageio.v3 as iio
import pywt
import skimage.color 
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Dictionary to hold different histogram components (if needed)
HistogramComponents = {}


def TextureExtractorFactory(type_str: str, histogram_bins: int = 256):

    parts = type_str.split('_')
    
    if "PiecewiseBlockLBP" in type_str:
        # Format: BlockLBP_<number_edge_block>_<R>_<P>
        number_edge_block = int(parts[1]) if len(parts) > 1 else 4
        R = float(parts[2]) if len(parts) > 2 else 1
        P = int(parts[3]) if len(parts) > 3 else 8
        return PiecewiseBlockLBPExtractor(number_edge_block=number_edge_block, R=R, P=P, histogram_bins=histogram_bins)

    if "BlockLBP" in type_str:
        # Format: BlockLBP_<number_edge_block>_<R>_<P>
        number_edge_block = int(parts[1]) if len(parts) > 1 else 4
        R = float(parts[2]) if len(parts) > 2 else 1
        P = int(parts[3]) if len(parts) > 3 else 8
        return BlockLBPExtractor(number_edge_block=number_edge_block, R=R, P=P, histogram_bins=histogram_bins)

    if "LBP" in type_str:
        # Format: LBP_<R>_<P>
        R = float(parts[1]) if len(parts) > 1 else 1
        P = int(parts[2]) if len(parts) > 2 else 8
        return LBPExtractor(R=R, P=P, histogram_bins=histogram_bins)
    
    
    elif "DCTConcat" in type_str:
        # Format: Texture-DCT_<block_fraction>_<coef_reduction_fraction>-0
        # fraction means block size = image size / block_fraction 
        # coef_reduction_fraction: in the slide
        _, block_fraction, coef_reduction_fraction, coef_normalize = type_str.split('_') 
        extractor = DCTConcatExtractor(block_fraction = int(block_fraction),
                                 coef_reduction_fraction = float(coef_reduction_fraction),
                                 coef_normalize = bool(coef_normalize))
        return extractor
    elif "DCTPiecewise" in type_str:
        # Format: Texture-DCT_<block_fraction>_<coef_reduction_fraction>-0
        # fraction means block size = image size / block_fraction 
        # coef_reduction_fraction: in the slide
        _, block_fraction, coef_reduction_fraction, coef_normalize = type_str.split('_') 
        extractor = DCTPiecewiseExtractor(block_fraction = int(block_fraction),
                                 coef_reduction_fraction = float(coef_reduction_fraction),
                                 coef_normalize = bool(coef_normalize))
        return extractor

    elif "BlockWavelet" in type_str:
        # Format: BlockWavelet_<number_edge_block>_<wavelet>_<levels>
        number_edge_block = int(parts[1]) if len(parts) > 1 else 3
        wavelet = parts[2] if len(parts) > 2 else 'haar'
        levels = int(parts[3]) if len(parts) > 3 else 3
        return BlockWaveletExtractor(number_edge_block=number_edge_block, wavelet=wavelet, levels=levels)

    elif "Wavelet" in type_str:
        # Format: Wavelet_<wavelet>_<levels>
        wavelet = parts[1] if len(parts) > 1 else 'haar'
        levels = int(parts[2]) if len(parts) > 2 else 3
        return WaveletExtractor(wavelet=wavelet, levels=levels)

    
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
    def __init__(self, R=1, P=8,histogram_bins:int = 256):
        super(LBPExtractor, self).__init__()
        self.R = R  # Radius of the LBP neighborhood
        self.P = P  # Number of points (neighbors) in the LBP neighborhood
        self.histogram_bins = histogram_bins

    def extract(self, image, normalize=True):
        if len(image.shape) == 3:
            gray_image = skimage.color.rgb2gray(image)
        else:
            gray_image = image
        gray_image = (gray_image * 255).astype(np.uint8)

        
        lbp = local_binary_pattern(gray_image, self.P, self.R, method='uniform')
        
        # plt.imshow(lbp, cmap='gray')
        # plt.show() bin_edges = np.linspace(0, P + 1, num=P + 2)
        bin_edges = np.linspace(0, self.P + 1, num=self.P + 2)
        histogram, _ = np.histogram(lbp.flatten(),bins=bin_edges)
        histogram = histogram / histogram.sum() if normalize else histogram

        return histogram

class PiecewiseBlockLBPExtractor(TextureExtractor):
    def __init__(self, number_edge_block: int = 4, R: int = 1, P: int = 8, histogram_bins: int = 256):
        super(PiecewiseBlockLBPExtractor, self).__init__()
        self.number_edge_block = number_edge_block
        self.R = R
        self.P = P
        self.histogram_bins = histogram_bins
    
    def extract(self, image, normalize=True):
        sizei = image.shape[0]
        sizej = image.shape[1]

        sizei_block = int(sizei / self.number_edge_block)
        sizej_block = int(sizej / self.number_edge_block)

        histograms = []  # List to hold histograms for each block
        
        # Loop through each block
        for i in range(self.number_edge_block):
            for j in range(self.number_edge_block):
                i_left_bound = (sizei_block) * i
                i_right_bound = (sizei_block * (i + 1))
                j_up_bound = (sizej_block) * j
                j_down_bound = (sizej_block * (j + 1))
                
                block_image = image[i_left_bound:i_right_bound, j_up_bound:j_down_bound, :]
                hist = TextureExtractorFactory(f'LBP_{self.R}_{self.P}', self.histogram_bins)
                
                sub_hist = hist.extract(block_image, normalize)
                histograms.append(sub_hist)  

        return histograms 


class DCTExtractor(TextureExtractor):
    def __init__(self, block_fraction: int = 16, coef_reduction_fraction: float = 0.5, coef_normalize: bool=False):
        super(DCTExtractor, self).__init__(histogram_bins=None)
        self.block_fraction = block_fraction
        self.coef_reduction_fraction = coef_reduction_fraction
        self.coef_normalize = coef_normalize
    def zigzag_scan(self, block, coef_reduction_fraction):
        h, w = block.shape
        result = []

        for sum in range(h + w - 1):
            if sum % 2 == 0:
                for i in range(sum + 1):
                    j = sum - i
                    if i < h and j < w:
                        result.append(block[j, i])
            else:
                for i in range(sum + 1):
                    j = sum - i
                    if i < h and j < w:
                        result.append(block[i, j])
        total_coeffs = len(result)
        num_to_keep = int(total_coeffs * coef_reduction_fraction)
        return result[:num_to_keep]

class DCTConcatExtractor(DCTExtractor):
    def __init__(self, block_fraction: int = 16, coef_reduction_fraction: float = 0.5, coef_normalize: int=0):
        super(DCTConcatExtractor, self).__init__(block_fraction, coef_reduction_fraction, coef_normalize)

    def extract(self, image):
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        fixed_size = 512
        gray_image = cv2.resize(gray_image, (fixed_size,fixed_size), interpolation=cv2.INTER_AREA)
        height, width = gray_image.shape
        
        block_size = fixed_size // self.block_fraction
        block_size = max(block_size, 8) #min: 8, not too small block
        if block_size % 2 != 0:
            block_size += 1

        dct_blocks = []
        for count_i,i in enumerate(range(0, height, block_size)):
            for count_j,j in enumerate(range(0, width, block_size)):
                
                block = gray_image[i:i+block_size, j:j+block_size]
                if block.shape[0] == block_size and block.shape[1] == block_size:
                    # DCT 
                    dct_block = cv2.dct(np.float32(block))
                    # zigzag scan
                    zigzag = self.zigzag_scan(dct_block, coef_reduction_fraction = self.coef_reduction_fraction)
                    dct_blocks.extend(zigzag)

        dct_coefficients = np.array(dct_blocks)
        if self.coef_normalize:
            dct_coefficients = (dct_coefficients - np.min(dct_coefficients)) / (np.max(dct_coefficients) - np.min(dct_coefficients))

        return [dct_coefficients]

class DCTPiecewiseExtractor(DCTExtractor):
    def __init__(self, block_fraction: int = 16, coef_reduction_fraction: float = 0.5, coef_normalize: int=0):
        super(DCTPiecewiseExtractor, self).__init__(block_fraction, coef_reduction_fraction, coef_normalize)

    def extract(self, image):
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        fixed_size = 512
        gray_image = cv2.resize(gray_image, (fixed_size,fixed_size), interpolation=cv2.INTER_AREA)
        height, width = gray_image.shape
        
        block_size = fixed_size // self.block_fraction
        block_size = max(block_size, 8) #min: 8, not too small block
        if block_size % 2 != 0:
            block_size += 1

        dct_blocks = []
        for count_i,i in enumerate(range(0, height, block_size)):
            for count_j,j in enumerate(range(0, width, block_size)):
                
                block = gray_image[i:i+block_size, j:j+block_size]
                if block.shape[0] == block_size and block.shape[1] == block_size:
                    # DCT 
                    dct_block = cv2.dct(np.float32(block))
                    # zigzag scan
                    zigzag = self.zigzag_scan(dct_block, coef_reduction_fraction = self.coef_reduction_fraction)
                    if self.coef_normalize:
                        zigzag = (zigzag - np.min(zigzag)) / (np.max(zigzag) - np.min(zigzag))
                    dct_blocks.append(zigzag)
        
        return dct_blocks
    
class BlockLBPExtractor(TextureExtractor):
    def __init__(self, number_edge_block: int = 4, R: int = 1, P: int = 8, histogram_bins: int = 256):
        super(BlockLBPExtractor, self).__init__()
        self.number_edge_block = number_edge_block
        self.R = R
        self.P = P
        self.histogram_bins = histogram_bins
    
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
        # print(self.R,self.P,self.number_edge_block)
        hist = TextureExtractorFactory(f'LBP_{self.R}_{self.P}', self.histogram_bins)
        
        hist_vector = np.zeros(0)
        for subimage in image_vectors:
            sub_hist = hist.extract(subimage, normalize)
            hist_vector = np.concatenate((hist_vector, sub_hist))

        return hist_vector


#possible wavelets
# haar, db2, sym3, bior1.3
class WaveletExtractor(TextureExtractor):
    def __init__(self, wavelet: str = 'haar', levels: int = 3):
        super(WaveletExtractor, self).__init__(0)
        self.wavelet = wavelet  
        self.levels = levels  # Number of decomposition levels

    def extract(self, image, normalize=True):
        # Convert the image to grayscale if it's not already
        fixed_size = 256
        if len(image.shape) == 3:
            gray_image = skimage.color.rgb2gray(image)
        else:
            gray_image = image
        gray_image = cv2.resize(gray_image, (fixed_size,fixed_size), interpolation=cv2.INTER_AREA)
        coeffs = pywt.wavedec2(gray_image, self.wavelet, level=self.levels)

        # Extract the detail coefficients (LH, HL, HH)
        features = []
        # Skip the first entry (LL band)
        for i in range(1, len(coeffs)):
            cH, cV, cD = coeffs[i]
            for sub_band in [cH, cV, cD]:
                # histogram, _ = np.histogram(sub_band, bins=self.histogram_bins, range=(sub_band.min(), sub_band.max()))
                # histogram = histogram / histogram.sum() if normalize else histogram
                features.extend(np.array(sub_band))

        return np.array(features).flatten()

class BlockWaveletExtractor(TextureExtractor):
    def __init__(self, number_edge_block: int = 3, wavelet: str = 'haar', levels: int = 1):
        super(BlockWaveletExtractor, self).__init__(0)
        self.number_edge_block = number_edge_block
        self.wavelet = wavelet
        self.levels = levels

    def extract(self, image, normalize=True):
        if len(image.shape) == 3:
            gray_image = skimage.color.rgb2gray(image)
        else:
            gray_image = image
        gray_image = cv2.resize(gray_image, (256,256))
        sizei = gray_image.shape[0]
        sizej = gray_image.shape[1]

        sizei_block = int(sizei / self.number_edge_block)
        sizej_block = int(sizej / self.number_edge_block)

        features = []
        for i in range(self.number_edge_block):
            for j in range(self.number_edge_block):
                i_left_bound = sizei_block * i
                i_right_bound = sizei_block * (i + 1)
                j_up_bound = sizej_block * j
                j_down_bound = sizej_block * (j + 1)
                
                block = gray_image[i_left_bound:i_right_bound, j_up_bound:j_down_bound]
                
                coeffs = pywt.wavedec2(block, self.wavelet, level=self.levels)
                
                # Extract coefficients (LH, HL, HH)
                for k in range(1, len(coeffs)):
                    cH, cV, cD = coeffs[k]
                    for sub_band in [cH, cV, cD]:
                        features.extend(sub_band)

        return np.array(features).flatten()




if __name__ == "__main__":

    image = iio.imread('target/BBDD/bbdd_00003.jpg')


    extractor = TextureExtractorFactory("PiecewiseBlockLBP_4_1.2_5",64)
    # extractor = TextureExtractorFactory("DCTPiecewise_16_0.1_0",64)


    features = extractor.extract(image)
    print(image.shape)
    # print(features)
    # print(features[1])
    # print(features.shape)

    print(len(features))
    print(len(features[1]))
    print(type(features))


