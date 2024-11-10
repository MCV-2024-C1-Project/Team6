import subprocess
import sys

# Hardcoded parameter ranges for KAZE
extended = [0, 1]  # 0 for False, 1 for True
threshold = [0.001, 0.005, 0.01, 0.02]
nOctaves = [3, 4, 5,6]
nOctaveLayers = [ 4]

# Prepare an output file to save the results
output_file = 'kaze_descriptor_results.txt'

# Function to generate KAZE descriptors and run predictions
def generate_and_predict(extended, threshold, nOctaves, nOctaveLayers, output_file='kaze_descriptor_results.txt'):
    with open(output_file, 'w') as f:
        # Redirect stdout to the file
        original_stdout = sys.stdout  # Store original stdout
        sys.stdout = f  # Redirect stdout to the file

        # Generate all descriptors with the `init` command
        print("Initializing descriptors...")
        for ext in extended:
            for thres in threshold:
                for nOct in nOctaves:
                    for nOctLayer in nOctaveLayers:
                        descriptor = f"KAZE_{ext}_{thres}_{nOct}_{nOctLayer}-0"
                        print(f"Generating descriptor: {descriptor}")
                        command = [
                            "C:/Python312/python.exe", "main.py", "init",
                            "--db_path", "target/BBDD",
                            "--descriptor-type", f"LocalFeat-{descriptor}"
                        ]
                        # subprocess.run(command)  # Uncomment to run the command

                        # # Write descriptor to output
                        # f.write(f"{descriptor}\n")

        print("Descriptor generation completed.")

        # Run predictions with the generated descriptors using the `predict` command
        print("Running predictions...")
        for ext in extended:
            for thres in threshold:
                for nOct in nOctaves:
                    for nOctLayer in nOctaveLayers:
                        descriptor = f"KAZE_{ext}_{thres}_{nOct}_{nOctLayer}-0"
                        print(f"Running prediction for descriptor: {descriptor}")
                        command = [
                            "C:/Python312/python.exe", "main.py", "predict",
                            "--input", "target/qsd1_w3",
                            "--result-number", "5",
                            "--descriptor-type", f"LocalFeat-{descriptor}",
                            "--measure", "L2",  # Use L2 for KAZE
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
    generate_and_predict(extended, threshold, nOctaves, nOctaveLayers)
