# descriptiors
import os
import imageio
import skimage.color 
import numpy as np
import pickle

def compute_descriptors(input_folder,output_folder):

    #setup the type of histogram you want
    num_bins = 20

    #list of picture name and its histogram
    descriptor_list = []

    image_paths = get_all_jpg_images(input_folder)

    for path in image_paths:
        image = imageio.imread(path)
        gray_image = skimage.color.rgb2gray(image)

        histogram, bin_edges = np.histogram(gray_image.flatten(),bins=num_bins)
        descriptor_list.append( [path,histogram] )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    descriptor_name = f"descriptor_grayscale_{num_bins}_bins.pkl"
    with open(output_folder+"/"+descriptor_name, "wb") as f:
        pickle.dump(descriptor_list,f)
        
    print(f"List saved to {output_folder}")
            

#retrieve all image paths from a folder
def get_all_jpg_images(input_folder):
    image_paths = []
    for file in  os.listdir(input_folder):
        if file.endswith(".jpg"):
            path = os.path.join(input_folder,file)
            image_paths.append(path)
    return image_paths