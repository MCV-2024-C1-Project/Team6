import os
import sys
import numpy as np
import pickle

def load_ground_truth(path):
    with open( os.path.abspath(path) ,"rb") as f:
        gt_list = pickle.load(f)
    return gt_list

def get_image_id(name):
    return int(name.split(".")[0])

def compute_APK(ground_truth, list_images):
    total_relevant_images = 0
    precisionK = 0
    for i, image_id in enumerate(images):
        if image_id in ground_truth:
            total_relevant_images += 1
            tmp = np.divide(total_relevant_images, i+1)
            precisionK =  np.sum(precisionK, tmp)
    
def compute_performance(result, number_queries, ground_truth_path):
    #inputr result format:
    #[((path_query_image, descriptor_query_image),list_Kresults) ... ]
    #sum(APk)q/Q
    Kpaths_list = [[ get_image_id(os.path.basename(os.path.abspath(path_result))) for (path_result,_,_) in lk_result] for (_,_), lk_result in result]  #[(path_result_image, metric, descriptor_result_image) ... ]
    gt_list = [{gt for gt in gt_arr} for gt_arr in load_ground_truth(ground_truth_path)] # a set of correct response for each query
    
    mean_apk = 0
    for i, gt in enumerate(gt_list):
        #gt is a set for generalization reason (in the future more than 1 image can be correct)
        mean_apk = np.sum(compute_APK(gt,Kpaths_list[i]))
    
    mean_apk = np.divide(mean_apk, len(gt_list))
    return mean_apk