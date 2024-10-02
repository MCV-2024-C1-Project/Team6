# descriptiors
import os
import imageio as iio
import skimage.color 
import numpy as np
import pickle

from src.feature import HistogramExtractor

def compute_descriptors(input_folder,output_folder):

    #setup the type of histogram you want
    num_bins = 256
    color_mode = 'GRAY'
    # color_mode = 'RGB'

    #list of picture name and its histogram
    descriptor_list = []

    image_paths = get_all_jpg_images(input_folder)
    hist_extractor = HistogramExtractor(color_mode= color_mode, histogram_bins= num_bins)

    for path in image_paths:
        histograms = hist_extractor.extract(path)
        descriptor_list.append( [path,histograms] )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    descriptor_name = f"descriptor_{color_mode.lower()}scale_{num_bins}_bins.pkl"
    with open(output_folder+"/"+descriptor_name, "wb") as f:
        pickle.dump(descriptor_list,f)
        
    print(f"List saved to {output_folder+'/'+descriptor_name}")
            
#retrieve pairs of [photo_path,descriptor] from pkl file
def load_descriptors(input_folder):
    with open(input_folder,"rb") as f:
        descriptor_list = pickle.load(f)
    return descriptor_list


#retrieve all image paths from a folder
def get_all_jpg_images(input_folder):
    image_paths = []
    for file in  os.listdir(input_folder):
        if file.endswith(".jpg"):
            path = input_folder+"/"+file
            image_paths.append(path)
    return image_paths