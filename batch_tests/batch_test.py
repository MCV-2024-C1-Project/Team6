import subprocess
import sys

# Hardcoded parameter ranges
nfeatures = [100, 500, 1000]
nOctaveLayers = [2, 3, 4, 5, 6]
contrastThreshold = [0.01, 0.03, 0.05, 0.08]

# Prepare an output file to save the results
output_file = 'sift_descriptor_results.txt'

# Function to generate descriptors and run predictions
def generate_and_predict(nfeatures, nOctaveLayers, contrastThreshold, output_file='sift_descriptor_results.txt'):
    with open(output_file, 'w') as f:
        # Redirect stdout to the file
        original_stdout = sys.stdout  # Store original stdout
        sys.stdout = f  # Redirect stdout to the file

        # Generate all descriptors with the `init` command
        print("Initializing descriptors...")
        for nfeature in nfeatures:
            for nOctaveLayer in nOctaveLayers:
                for contrastThres in contrastThreshold:
                    descriptor = f"SIFT_{nfeature}_{nOctaveLayer}_{contrastThres}-0"
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
            for nOctaveLayer in nOctaveLayers:
                for contrastThres in contrastThreshold:
                    descriptor = f"SIFT_{nfeature}_{nOctaveLayer}_{contrastThres}-0"
                    print(f"Running prediction for descriptor: {descriptor}")
                    command = [
                        "C:/Python312/python.exe", "main.py", "predict", 
                        "--input", "target/qsd1_w3", 
                        "--result-number", "5", 
                        "--descriptor-type", f"LocalFeat-{descriptor}", 
                        "--measure", "L2", 
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
    # Generate descriptors and run predictions
    generate_and_predict(nfeatures, nOctaveLayers, contrastThreshold)
