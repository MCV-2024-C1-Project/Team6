import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def plot_result(input, score_list, feature_names, apk=-1):
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
    fig, axes = plt.subplots(num_rows, num_cols, constrained_layout=True)


    axes[0, 0].imshow(input_image)
    axes[0, 0].set_title(f"Input Image: {input_image_path}{string_score}")
    axes[0, 0].axis('off')

    for col in range(1, num_cols):
        x = np.arange(len(input_feature[col - 1]))
        axes[0, col].plot(x, input_feature[col - 1], color='blue')
        axes[0, col].fill_between(x, input_feature[col - 1], color='blue', alpha=0.3)
        axes[0, col].set_title(f"{feature_names[col-1]} - Input Image")
        axes[0, col].set_xticks([])

    for i, (sim_path, sim_image, sim_metric) in enumerate(zip(similar_paths, similar_images, similar_scores)):
        
        axes[i+1, 0].imshow(sim_image)
        axes[i+1, 0].set_title(f"{i}: {sim_path.split('//')[1]} | Distance: {sim_metric:.2f}")
        axes[i+1, 0].axis('off')

        for col in range(1, num_cols):
            x = np.arange(len(similar_features[i][col - 1]))
            axes[i + 1, col].plot(x, similar_features[i][col - 1], color='orange')
            axes[i + 1, col].fill_between(x, similar_features[i][col - 1], color='orange', alpha=0.3)
            axes[i + 1, col].set_title(f"{feature_names[col-1]} - Similar Image {i + 1}")
            axes[i + 1, col].set_xticks([])

    # plt.tight_layout()
    # plt.subplots_adjust(hspace=0.6)  # Increase horizontal (wspace) and vertical (hspace) padding
    plt.show()
