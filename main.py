import sys
import argparse
import os
import csv
import math
import src.descriptors as descriptors
import src.query as query
from src.plotting import ImageNavigator
from src.histogram import HistogramComponents
import re
import pickle

def image_name_to_id(image_name):
    match = re.search(r'(\d+)', image_name)
    if match:
        return int(match.group(1))
    return -1

def init_command(args):
    descriptor_type, descriptor_subtype, num_bins = args.descriptor_type.split("-")
    descriptors.generate_descriptors_DBfile(args.db_path, "generated_descriptors", descriptor_type, descriptor_subtype, num_bins=int(num_bins))
    print("Database initialized.")

def predict_command(args):
    descriptor_type, descriptor_subtype, num_bins = args.descriptor_type.split("-")
    single_image = os.path.isfile(args.input)
    result = query.prediction(args.input, 
                              "generated_descriptors", 
                              int(args.result_number), 
                              descriptor_type, 
                              descriptor_subtype, 
                              int(num_bins),
                              args.measure,
                              single_image=single_image, 
                              evaluate=args.evaluate) # mapk, apk, (path, dist, raw_dist, db_image_descriptor)
    feature_ids = HistogramComponents[descriptor_subtype]
    output_result = []
    plotting_results = []
    if args.evaluate:
        score, apk_list, result_list = result
        for i, (query_input, score_list) in enumerate(result_list):
            result_names = [s[0] for s in score_list]
            output_result.append([image_name_to_id(name) for name in result_names])
            print(f"{query_input[0]} ==> {result_names} | Performance(AP@K): {apk_list[i]}")
            plotting_results.append((query_input, score_list, apk_list[i]))
        print(f"Performance score (MAP@K): {score}")
    else:
        result_list = result
        for query_input, score_list in result_list:
            result_names = [s[0] for s in score_list]
            output_result.append([image_name_to_id(name) for name in result_names])
            print(f"{query_input[0]} ==> {result_names}")
            plotting_results.append((query_input, score_list, None))
            
    if args.save_output:
        output_path = args.output
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        print(output_result)
        with open(os.path.join(output_path, "result.pkl"), "wb") as f:
            pickle.dump(output_result, f)
    if args.plot:
        navigator = ImageNavigator(plotting_results, feature_ids)
        navigator.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image processing tool with different subcommands')
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Subparser for the 'init' command
    init_parser = subparsers.add_parser('init', help='Initialize the database')
    init_parser.add_argument('--db_path', required=False, default="BBDD/", help='Path to the database')
    init_parser.add_argument('--descriptor-type', required=False, default="Histogram-RGB-256", help='Descriptor type to be used, in the format "histogram-color-bins"')
    init_parser.set_defaults(func=init_command)

    # Subparser for the 'predict' command
    predict_parser = subparsers.add_parser('predict', help='Predict results for a single image')
    predict_parser.add_argument('--input', required=False, default="qsd1_w1/", help='Input image or folder')
    predict_parser.add_argument('--result-number', required=False, default=1, help='Number of results to return, aka K')
    predict_parser.add_argument('--descriptor-type', required=False, default="Histogram-Super-64", help='Descriptor type to be used, in the format "histogram-color-bins"')
    predict_parser.add_argument('--measure', required=False, default="HellingerKernel-Median", help='Measure function to be used for similarity ranking')
    predict_parser.add_argument('--plot', action='store_true', help='If set, show result plots')
    predict_parser.add_argument('--evaluate', action='store_true', help='If set, perform evaluation using ground truth')
    predict_parser.add_argument('--save-output', action='store_true', help='If set, save the prediction results to a CSV file')
    predict_parser.add_argument('--output', required=False, default="results/", help='Directory to save output files if --save-output is set')
    predict_parser.set_defaults(func=predict_command)

    # Parse the arguments and call the appropriate function
    args = parser.parse_args()
    args.func(args)