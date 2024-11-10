import os
import sys
import numpy as np
import pickle

def load_ground_truth(path):
    with open( os.path.abspath(path) ,"rb") as f:
        gt_list = pickle.load(f)
    return gt_list

def get_image_id(name):
    return int((name.split(".")[0]).split("bbdd_")[1])

def compute_APK(ground_truth, list_images):
    
    total_relevant_images = 0
    precisionK = 0

    for i, image_id in enumerate(list_images):
        if image_id in ground_truth:
            total_relevant_images += 1
            tmp = np.divide(total_relevant_images, i+1)
            precisionK =  precisionK + tmp
            
    if total_relevant_images == 0:
        return 0
    else:
        return np.divide(precisionK,total_relevant_images)
    
def old_compute_performance(result, ground_truth_path, debug=False):
    #inputr result format:
    #[((path_query_image, descriptor_query_image),list_Kresults) ... ]
    #sum(APk)q/Q
    Kpaths_list = [[ get_image_id(os.path.basename(os.path.abspath(path_result))) for (path_result,_,_,_) in lk_result] for (_,_), lk_result in result]  #[(path_result_image, metric, descriptor_result_image) ... ]
    print(load_ground_truth(ground_truth_path))
    gt_list = [{gt for gt in gt_arr} for gt_arr in load_ground_truth(ground_truth_path)] # a set of correct response for each query
    list_apk = []
    mean_apk = 0
    for i, gt in enumerate(gt_list):
        #gt is a set for generalization reason (in the future more than 1 image can be correct)
        apk = compute_APK(gt, Kpaths_list[i])
        list_apk.append(apk)
        mean_apk = mean_apk + apk
        if debug:
            print(f"APK for query im {str(gt)}: {apk}")
            print(f"gt: {str(gt)}// K results: {str(Kpaths_list[i])}")
            print("-----------")
        #####
    mean_apk = np.divide(mean_apk, len(gt_list))
    return mean_apk, list_apk

def compute_performance(result, ground_truth_path, debug=True):
    query_result_bbdd_id = [] #len(query_result_bbdd_id) == 30, image in lk_paintings_image, image list of N elements (N is paintings on image), each painting in image, painting is a list of K result(k bbdd_id)
    for image_result in result:
        image_result_bbdd_id = []
        for (_,_), painting_result_lk in image_result:
            if len(painting_result_lk) == 1 and painting_result_lk[0][0] == "-1" and painting_result_lk[0][1] == np.inf:
                image_result_bbdd_id.append([-1])
            else:
                lk_painting_bbdd_id = []
                for (path_result,_,_,_) in painting_result_lk:
                    bbdd_id = get_image_id(os.path.basename(os.path.abspath(path_result)))
                    lk_painting_bbdd_id.append(bbdd_id)
                image_result_bbdd_id.append(lk_painting_bbdd_id)

        query_result_bbdd_id.append(image_result_bbdd_id)

    print(load_ground_truth(ground_truth_path))
    gt_list = [[{gt} for gt in gt_arr] for gt_arr in load_ground_truth(ground_truth_path)] #array with gt for each image of gt (so a list of sets of one element, which can be 1 or 2 ass they can be 1 or 2 paitings in an image)
    list_apk = []
    mean_apk = 0
    total_paiting_query = 0
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for i_image, gt_image in enumerate(gt_list):
        for i_painting, gt_painting in enumerate(gt_image):
            total_paiting_query += 1
            if (i_painting >= len(query_result_bbdd_id[i_image]) ):
                print(f"We predicted {str(len(query_result_bbdd_id[i_image]))} paintings for image {str(i_image)}, however the gt says they are {str(len(gt_image))} paintings")
                list_apk.append(0)
                continue
            apk = compute_APK(gt_painting, query_result_bbdd_id[i_image][i_painting])
            if -1 in gt_painting:
                if len(query_result_bbdd_id[i_image][i_painting]) == 1 and query_result_bbdd_id[i_image][i_painting][0] == -1:
                    TP += 1
                else:
                    FN += 1
            else:
                if len(query_result_bbdd_id[i_image][i_painting]) == 1 and query_result_bbdd_id[i_image][i_painting][0] == -1:
                    FP += 1
                else:
                    TN += 1

            list_apk.append(apk)
            mean_apk = mean_apk + apk
            if debug:
                print(f"APK for query im {i_image} and paiting {i_painting} {str(gt_painting)}: {apk}")
                print(f"gt: {str(gt_painting)}// K results: {str(query_result_bbdd_id[i_image][i_painting])}")
                print("-----------")
            #####
    mean_apk = np.divide(mean_apk, total_paiting_query) if total_paiting_query > 0 else 0
    if debug:
        print(f"mean apk: {mean_apk}")
        # print(f"TP:{TP}, FN:{FN}, FP:{FP}, TN:{TN}")
        # print(f"F1_Score for -1 lists:{2*TP/(2*TP+FP+FN)}, precision:{ TP/(TP+FP)}, recall:{TP/(TP+FN)}")

    return mean_apk, list_apk