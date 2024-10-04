import imageio
import sys
import os
from skimage.color import rgb2gray
from src.descriptors import compute_descriptor
from src.descriptors import get_all_jpg_images
from src.descriptors import load_descriptors
from src.measures import MeasureFactory
from src.performance import compute_performance

def retrieve_K(descriptor, db_descriptor, k):
    measure = MeasureFactory("HellingerKernel")
    # first interation can  been improved O(n²) depending of sort implementation
    result = [(path, measure(descriptor,db_image_descriptor), db_image_descriptor) for path, db_image_descriptor in db_descriptor]
    result = sorted(result, key=lambda x: x[1]) # we sort the list by using a function that extract the score
    return result[:k] # return the first k element of the list

def prediction(input_path, db_path, k, descriptor_type, descriptor_subtype, evaluate=False, single_image=False):
    db_descriptor = load_descriptors(db_path, descriptor_type, descriptor_subtype) # format: [(path1, descriptor1),(path2, descriptor2),...,(pathN, descriptorN)]
    image_paths = [input_path] if single_image else get_all_jpg_images(input_path)

    result = []

    for path in image_paths:
        descriptor = compute_descriptor(path, descriptor_type, descriptor_subtype)
        k_result = retrieve_K(descriptor, db_descriptor, k) # listKresults = [(path_result_image, metric, descriptor_result_image) ... ]
        result.append( ((path, descriptor), k_result) ) # [((path_query_image, descriptor_query_image),list_Kresults) ... ]
    
    if evaluate:
        mapk, list_apk = compute_performance(result, os.path.join(input_path, "gt_corresps.pkl"))
        return (mapk, list_apk, result)
    else:
        return result