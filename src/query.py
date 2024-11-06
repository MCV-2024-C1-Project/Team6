import imageio
import sys
import os
import numpy as np
import cv2
from skimage.color import rgb2gray
from src.descriptors import compute_descriptor
from src.descriptors import get_all_jpg_images
from src.descriptors import load_descriptors
from src.measures import MeasureFactory
from src.measures import LocalFeatMeasureExtractor
from src.performance import compute_performance
from src.background_remover import crop_foreground
from src.background_remover2 import frame_detector
from src.denoising import noise_removal
import matplotlib.pyplot as plt
from tqdm import tqdm

def compare_localfeats(vector_local_feat, db_image_descriptor, score_sys, point_threshold=np.inf, rate_treshold=-1, alpha=0.5):
   
    measure = MeasureFactory(measure)
    result = np.array([None for x in range(len(vector_local_feat))])
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(vector_local_feat,des2,k=2)
    
    for i, localfeat_point in enumerate(vector_local_feat):
        dist_point_vector = []
        for j, localfeat_point_db in enumerate(db_image_descriptor)
            dist = measure(localfeat_point, localfeat_point_db)
            dist_point_vector.append((dist, j))
        dist_point_vector = sorted(dist_point_vector, key=lambda x: x[0])
        if dist_point_vector[0][0] <= point_threshold and abs(dist_point_vector[0][0]-dist_point_vector[1][0]) > rate_treshold
            result[i] = dist_point_vector[0]
    
    valid_distances = np.array([x[0] for x in result if not(x is None)])
    if len(valid_distances) == 0:
        return np.inf
    if score_sys == "match":
        return 1.0 - len(valid_distances)/len(result)#number_matches/total_points
    elif score_sys == "avg":
        return sum(valid_distances)/len(valid_distances)
    elif score_sys == "weighted":
        match_rate = 1.0 - len(valid_distances)/len(result)
        average_dist = sum(valid_distances)/len(valid_distances)
        return average_dist*match_rate
    else:
        sys.exit("ERROR: Invalid method for computing whole distance for a local feature")
    #TODO: maybe ensure one unique asigment for each point?
    # optimizatio option: flann matcher
            
            

def retrieve_K_localfeat(descriptor, db_descriptor, measure, k, type_local_feat="SIFT", non_existent_behavior=False, score_sys="match", treshold=0.5):
    
    result = []
    #descriptor is a vector of local features, one for each keypoint detected
    for path, db_image_descriptor in db_descriptor:
        dist = compare_localfeats(descriptor, db_image_descriptor, score_sys)
        result.append((path, dist, db_image_descriptor, []))

    result = sorted(result, key=lambda x: x[1])
    result = [x for x in result if not(x[1] == np.inf)]
    if len(result) == 0 :
        #if all the db images have a distance of inf
        return [("-1", np.inf, [], [])]
    
    return result[:k] 

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
            if descriptor_type == "LocalFeat":
                k_result = retrieve_K_localfeat(descriptor, db_descriptor, measure, k) # listKresults = [(path_result_image, metric, descriptor_result_image, raw_metric) ... ]
            else:
                k_result = retrieve_K(descriptor, db_descriptor, measure, k) # listKresults = [(path_result_image, metric, descriptor_result_image, raw_metric) ... ]
            result_tmp.append( ((path, descriptor), k_result) ) # [((path_query_image, descriptor_query_image),list_Kresults) ... ] 
        result.append(result_tmp) # now result is an array where each element is a vector of N elements (N paitings 1 image). So these an element of result vector[1 paiting result  image 1,..., N paiting result  image 1]
    if evaluate:
        mapk, list_apk = compute_performance(result, os.path.join(input_path, "gt_corresps.pkl")) # recheck for our implementation
        return (mapk, list_apk, result)
    else:
        return result