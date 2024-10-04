import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def plot_result(input, score_list, apk=-1):
    input_image_path = input[0]
    input_image = Image.open(input[0])
    input_feature = input[1]
    similar_paths = [f[0] for f in score_list]
    similar_images = [Image.open(f[0]) for f in score_list]
    similar_scores = [f[1] for f in score_list]
    similar_features = [f[2] for f in score_list]
    string_score = "" if apk < 0 else f"| Performance: {apk:.2f}"

    num_rows = len(score_list) + 1
    num_cols = 1+len(input_feature)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))


    axes[0, 0].imshow(input_image)
    axes[0, 0].set_title(f"Input Image: {input_image_path}{string_score}")
    axes[0, 0].axis('off')

    for col in range(1, num_cols):
        axes[0, col].bar(np.arange(len(input_feature[col-1])), input_feature[col-1])
        axes[0, col].set_title(f"Feature {col} - Input Image")
        axes[0, col].set_xticks([])

    for i, (sim_path, sim_image, sim_metric) in enumerate(zip(similar_paths, similar_images, similar_scores)):
        
        axes[i+1, 0].imshow(sim_image)
        axes[i+1, 0].set_title(f"1: {sim_path} | Distance: {sim_metric:.2f}")
        axes[i+1, 0].axis('off')

        for col in range(1, num_cols):
            axes[i+1, col].bar(np.arange(len(similar_features[i][col-1])), similar_features[i][col-1])
            axes[i+1, col].set_title(f"Feature {col} - Similar Image {i+1}")
            axes[i+1, col].set_xticks([])


    plt.tight_layout()
    plt.show()
