import os
import imageio.v3 as iio
import skimage.color 
import numpy as np

class HistogramExtractor(object):
    def __init__(self, color_mode:str = 'RGB', histogram_bins:int = 256):
        self.color_mode = color_mode
        self.histogram_bins = histogram_bins

    def extract(self, image_path:str, normalize = True):
        # Caution: image from imageio is RGB, from cv2 is RBG
        image = iio.imread(image_path)
        self.histograms = [] 
        if self.color_mode == 'RGB':
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

        elif self.color_mode == 'GRAY':
            #GRAY mode
            bin_edges = np.linspace(0, 1, num=self.histogram_bins + 1)
            gray_image = skimage.color.rgb2gray(image)
            histogram, bin_edges = np.histogram(
                gray_image.flatten(),
                bins=bin_edges
                )
            histogram = histogram / histogram.sum() if normalize else histogram
            self.histograms.append(histogram)

        else: # Placeholder
            pass
        
        return self.histograms
    
