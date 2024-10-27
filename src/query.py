import imageio
import sys
import os
import numpy as np
from skimage.color import rgb2gray
from src.descriptors import compute_descriptor
from src.descriptors import get_all_jpg_images
from src.descriptors import load_descriptors
from src.measures import MeasureFactory
from src.performance import compute_performance
from src.background_remover import crop_foreground
from src.background_remover2 import frame_detector
from src.denoising import noise_removal
import matplotlib.pyplot as plt
from tqdm import tqdm

def retrieve_K(descriptor, db_descriptor, measure, k):
    measure = MeasureFactory(measure)
    # first interation can  been improved O(n²) depending of sort implementation
    result = []
    for path, db_image_descriptor in db_descriptor:
        dist, raw_dist = measure(descriptor,db_image_descriptor)
        result.append((path, dist, db_image_descriptor, raw_dist))
    result = sorted(result, key=lambda x: x[1]) # we sort the list by using a function that extract the score
    return result[:k] # return the first k element of the list

def prediction(input_path, db_path, k, descriptor_type, descriptor_subtype, num_bins, measure, evaluate=False, single_image=False, remove_background=False, remove_noise=False, noise_filter_arguments=None):
    db_descriptor = load_descriptors(db_path, descriptor_type, descriptor_subtype, num_bins=num_bins) # format: [(path1, descriptor1),(path2, descriptor2),...,(pathN, descriptorN)]
    image_paths = [input_path] if single_image else get_all_jpg_images(input_path)

    result = []

    for path in tqdm(image_paths, desc="Processing images"):
        raw_image = imageio.imread(path)
        if remove_background:
            list_image = frame_detector(raw_image, False) # now crop image may generate more than one , ideally starting a top left and ending a bottom right
            # print(raw_image.dtype, np.max(raw_image))
            # print(list_image[0].dtype, np.max(list_image[0]))
            # _, ax = plt.subplots(1,len(list_image)+1)
            # ax[0].imshow(raw_image)
            # ax[0].axis('off')
            # for i in range(len(list_image)):
            #     ax[i+1].imshow(list_image[i])
            #     ax[i+1].axis('off')
            # plt.show()
        else:
            list_image = [raw_image]
        
        if remove_noise:
            list_image = [noise_removal(image, noise_filter_arguments) for image in list_image] 

        result_tmp = []
        for image in list_image:
            descriptor = compute_descriptor(image, descriptor_type, descriptor_subtype, num_bins=num_bins)
            k_result = retrieve_K(descriptor, db_descriptor, measure, k) # listKresults = [(path_result_image, metric, descriptor_result_image, raw_metric) ... ]
            result_tmp.append( ((path, descriptor), k_result) ) # [((path_query_image, descriptor_query_image),list_Kresults) ... ] 
        result.append(result_tmp) # now result is an array where each element is a vector of N elements (N paitings 1 image). So these an element of result vector[1 paiting result  image 1,..., N paiting result  image 1]
    if evaluate:
        mapk, list_apk = compute_performance(result, os.path.join(input_path, "gt_corresps.pkl"))
        return (mapk, list_apk, result)
    else:
        return result