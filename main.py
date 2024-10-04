import sys
import argparse
import os
import csv
import sys
import math
import src.descriptors as descriptors
import src.query as query
from src.plotting import plot_result


DATABASE_PATH = "./"


if __name__ == '__main__':
    query_folder = sys.argv[1]
    parser = argparse.ArgumentParser()
    
    #init: create file descriptor for database (input).
    #predict: predict images for a single input.
    #batch-predict: run predictions for all the files in the input folder.
    #evaluate: same as batch-predict but with precision metrics using ground truth assuming it's present in input folder.
    parser.add_argument('--action', 
                        required=True,
                        choices=["init","predict","batch-predict", "evaluate"],
                        default="", 
                        help='action to perform')
    
    parser.add_argument('--input', 
                        required=False,
                        default="qst1_w1/", 
                        help='input argument')
    
    
    parser.add_argument('--output', 
                        required=False, 
                        default="results/", 
                        help='Number of results')

    parser.add_argument('--result-number', 
                        required=False, 
                        default=1, 
                        help='Number of results to return, a.k.a K')
    
    parser.add_argument('--metric', 
                        required=False, 
                        default="HellingerKernel", 
                        help='Metric used to compute distance between images')
    
    parser.add_argument('--descriptor-type', 
                        required=False, 
                        default="Histogram-RGB", 
                        help='Descriptor type to be used, with the format "type-subtype"')

    parser.add_argument('--db-path', 
                        required=False, 
                        default="BBDD/", 
                        help='Descriptor type to be used, with the format "type-subtype"')
    
    args = parser.parse_args()

    action = args.action
    k = args.result_number
    descriptor_type = args.descriptor_type.split("-")[0]
    descriptor_subtype = args.descriptor_type.split("-")[1]

    if action == "init":
        descriptors.generate_descriptors_DBfile(args.db-path, "generated_descriptors", descriptor_type, descriptor_subtype)
        #descriptor_list = src.descriptors.load_descriptors("generated_descriptors/descriptor_grayscale_20_bins.pkl")
        #print(descriptor_list)
    
    elif action == "predict":
        result_list = query.prediction(args.input, "generated_descriptors", k, descriptor_type, descriptor_subtype, single_image=True) #returns a list of tuples, (path, metric)
        for query_input, score_list in result_list:
            print(f"Result for the first {k} images, for the query : {query_input[0]}")
            plot_result(query_input, score_list)

    elif action == "batch-predict":
        result_list = query.prediction(args.input, "generated_descriptors", k, descriptor_type, descriptor_subtype) #returns a list of tuples, each element being (input_query_image_path, list of tuples (path_resulting_image, metric))
        for query_input, score_list in result_list:
            print(f"Result for the first {k} images, for the query : {query_input[0]}")
            plot_result(query_input, score_list)

    elif action == "evaluate":
        pass

    else:
        pass
    # src.query.run_query_week1(args.path, args.outputdir)