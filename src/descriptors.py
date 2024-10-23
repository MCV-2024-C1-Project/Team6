# descriptors
import os
import sys
import imageio as iio
import skimage.color 
import numpy as np
import pickle

from src.histogram import HistogramExtractorFactory

def compute_descriptor(image, type, subtype, num_bins=256):  
    if type == "Histogram":
        #setup the type of histogram you want
        color_mode = subtype #GRAY or RGB from this moment
        hist_extractor = HistogramExtractorFactory(type = color_mode, histogram_bins = num_bins)
        return hist_extractor.extract(image)
    elif type == "Texture":
        hist_extractor = TextureExtractorFactory(type = color_mode, histogram_bins = num_bins)
    else:
        sys.exit("Not yet implemented")

#maybe change the name to save descriptors?
def generate_descriptors_DBfile(input_folder, output_folder, descriptor_type, descriptor_subtype, num_bins=256):

    image_paths = get_all_jpg_images(input_folder)
    descriptor_list = []
    for path in image_paths:
        raw_image = iio.imread(path)
        descriptor = compute_descriptor(raw_image, descriptor_type, descriptor_subtype, num_bins)
        descriptor_list.append( (path, descriptor) ) #changed from array to tuple, more easily iteratable

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    descriptor_name = f"descriptor_{descriptor_type.lower()}_{descriptor_subtype.lower()}_{num_bins}_bins.pkl"
    with open(output_folder+"/"+descriptor_name, "wb") as f:
        pickle.dump(descriptor_list,f)
        
    print(f"List saved to {output_folder+'/'+descriptor_name}")
            
#retrieve pairs of (photo_path, descriptor) from pkl file
def load_descriptors(input_folder, descriptor_type, descriptor_subtype, num_bins=256):
    descriptor_name = f"descriptor_{descriptor_type.lower()}_{descriptor_subtype.lower()}_{num_bins}_bins.pkl"
    with open(os.path.join(input_folder, descriptor_name),"rb") as f:
        descriptor_list = pickle.load(f)
    return descriptor_list


#retrieve all image paths from a folder
def get_all_jpg_images(input_folder):
    image_paths = []
    for file in  os.listdir(input_folder):
        if file.endswith(".jpg"):
            path = input_folder+"/"+file
            image_paths.append(path)

    return sorted(image_paths)

def get_list_descriptor(list_descriptor):
    pass