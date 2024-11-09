import subprocess
import sys

# Hardcoded parameter ranges for ORB
nfeatures = [100, 500, 1000]
WTA_K = [2, 3, 4]
fastThreshold = [10, 20, 30]

# Prepare an output file to save the results
output_file = 'orb_descriptor_results.txt'

# Function to generate descriptors and run predictions
def generate_and_predict(nfeatures, WTA_K, fastThreshold, output_file='orb_descriptor_results.txt'):
    with open(output_file, 'w') as f:
        # Redirect stdout to the file
        original_stdout = sys.stdout  # Store original stdout
        sys.stdout = f  # Redirect stdout to the file

        # Generate all descriptors with the `init` command
        print("Initializing descriptors...")
        for nfeature in nfeatures:
            for wta_k in WTA_K:
                for fast_thresh in fastThreshold:
                    descriptor = f"ORB_{nfeature}_{wta_k}_{fast_thresh}-0"
                    print(f"Generating descriptor: {descriptor}")
                    command = [
                        "C:/Python312/python.exe", "main.py", "init", 
                        "--db_path", "target/BBDD", 
                        "--descriptor-type", f"LocalFeat-{descriptor}"
                    ]
                    subprocess.run(command)  # Uncomment to run the command

                    # Write descriptor to output
                    f.write(f"{descriptor}\n")

        print("Descriptor generation completed.")

        # Run predictions with the generated descriptors using the `predict` command
        print("Running predictions...")
        for nfeature in nfeatures:
            for wta_k in WTA_K:
                if wta_k == 2:
                    baka = "Hamming"
                else:
                    baka = "Hamming2"
                for fast_thresh in fastThreshold:
                    descriptor = f"ORB_{nfeature}_{wta_k}_{fast_thresh}-0"
                    print(f"Running prediction for descriptor: {descriptor}")
                    command = [
                        "C:/Python312/python.exe", "main.py", "predict", 
                        "--input", "target/qsd1_w3", 
                        "--result-number", "5", 
                        "--descriptor-type", f"LocalFeat-{descriptor}", 
                        "--measure", baka, 
                        "--evaluate",
                        "--remove-noise",
                        "--filter-type", "Median-3"
                    ]
                    subprocess.run(command)

                    # Assuming the prediction returns some results to log
                    # e.g., f.write(f"{descriptor}: {results}\n")  # Replace with actual results

        print("Prediction completed.")

        # Restore the original stdout
        sys.stdout = original_stdout

    print(f"All output has been saved to {output_file}.")

if __name__ == '__main__':
    # Generate descriptors and run predictions for ORB
    generate_and_predict(nfeatures, WTA_K, fastThreshold)
