import sys
import argparse
import os
import csv
import sys
import math
import src.query

DATABASE_PATH = "./"


if __name__ == '__main__':
    query_folder = sys.argv[1]
    parser = argparse.ArgumentParser()
    
    #init: create file descriprot for database (input).
    #predict: predict images for a single input.
    #batch-predict: run predictions for all the files in the input folder.
    #evaluate: same as batch-predict but with precision metrics using ground truth assuming it's present in input folder.
    parser.add_argument('--action', 
                        required=True,
                        choices=["init","predict","batch-predict", "evaluate"],
                        default="", 
                        help='action to perform')
    
    parser.add_argument('--input', 
                        required=True, 
                        default="", 
                        help='input argument')
    
    parser.add_argument('--result-number', 
                        required=False, 
                        default=1, 
                        help='Number of results')
    args = parser.parse_args()
    action = args.action
    if action == "init":
        pass
    elif action == "predict":
        src.query.single_prediction(args.input, '')
    elif action == "batch-predict":
        pass
    elif action == "evaluate":
        pass
    else:
        pass
    # src.query.run_query_week1(args.path, args.outputdir)