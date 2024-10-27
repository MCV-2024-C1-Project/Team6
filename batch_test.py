import subprocess
import sys

# Define the descriptor parameters as pairs of (R, P)
rp_pairs = [
    (1, 8),
    (2.5, 12),
    (4, 16)
]

blocks = [4, 8, 16, 24, 32]  # Number of blocks

# Prepare an output file to save the results
output_file = 'descriptor_results.txt'

# Step 1: Open the output file and redirect stdout
with open(output_file, 'w') as f:
    # Redirect stdout to the file
    original_stdout = sys.stdout  # Store original stdout
    sys.stdout = f  # Redirect stdout to the file

    # Generate all descriptors with the `init` command
    print("Initializing descriptors...")
    for block in blocks:
        for r, p in rp_pairs:
            descriptor = f"PiecewiseBlockLBP_{block}_{r}_{p}-64"
            print(f"Generating descriptor: {descriptor}")
            command = [
                "C:/Python312/python.exe", "main.py", "init", 
                "--db_path", "target/BBDD", 
                "--descriptor-type", f"Texture-{descriptor}"
            ]
            subprocess.run(command)

            # Write descriptor to output
            f.write(f"{descriptor}\n")

    print("Descriptor generation completed.")

    # Step 2: Run predictions with the generated descriptors using the `predict` command
    print("Running predictions...")
    for block in blocks:
        for r, p in rp_pairs:
            descriptor = f"PiecewiseBlockLBP_{block}_{r}_{p}-64"
            print(f"Running prediction for descriptor: {descriptor}")
            command = [
                "C:/Python312/python.exe", "main.py", "predict", 
                "--input", "target/qsd1_w1", 
                "--result-number", "5", 
                "--descriptor-type", f"Texture-{descriptor}", 
                "--measure", "L1-Median", 
                "--evaluate"
            ]
            subprocess.run(command)

            # Assuming the prediction returns some results to log
            # Append results to output file if needed
            # e.g., f.write(f"{descriptor}: {results}\n")  # Replace with actual results

    print("Prediction completed.")

    # Restore the original stdout
    sys.stdout = original_stdout

print(f"All output has been saved to {output_file}.")
