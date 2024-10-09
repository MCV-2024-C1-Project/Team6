import imageio
import matplotlib.pyplot as plt
import os
import numpy as np

folder_path = "target/qsd2_w1"
output_folder = "target/plots"


def remove_background(image):
    
    depth = 8
    K = 2
    sample_frequency = 50

    shape = image.shape

    color_samples = []

    #sample colors around all edges at given depth 
    step = int(np.floor(shape[0]/ sample_frequency))
    for j in range(depth, shape[0], step):
        color_samples.append(image[j,depth])
        color_samples.append(image[j,shape[1] - depth])

    step = int(np.floor(shape[1]/ sample_frequency))
    for j in range(depth, shape[1], step):
        color_samples.append(image[depth,j])
        color_samples.append(image[shape[0] - depth,j])


    color_samples = np.array(color_samples)

    R_samples = color_samples[:, 0]
    G_samples = color_samples[:, 1]
    B_samples = color_samples[:, 2]


    mean_color = [np.mean(R_samples), np.mean(G_samples), np.mean(B_samples)]
    std_color = [np.std(R_samples), np.std(G_samples), np.std(B_samples)]

    mask = ((image[:, :, 0] > mean_color[0] - K * std_color[0]) & (image[:, :, 0] < mean_color[0] + K * std_color[0]) &
            (image[:, :, 1] > mean_color[1] - K * std_color[1]) & (image[:, :, 1] < mean_color[1] + K * std_color[1]) &
            (image[:, :, 2] > mean_color[2] - K * std_color[2]) & (image[:, :, 2] < mean_color[2] + K * std_color[2])).astype('uint8')
    
    return mask









#Code for mass testing and plotting
images_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".jpg")]
batch_size = 10
num_batches = int(np.ceil(len(images_list) / batch_size))

for batch_index in range(num_batches):
    plt.figure(figsize=(6, 20)) 

    batch_images = images_list[batch_index * batch_size:(batch_index + 1) * batch_size]
    num_images_in_batch = len(batch_images)

    

    for i, image_path in enumerate(batch_images):

        image = imageio.imread(image_path)
        mask = remove_background(image)
  
        plt.subplot(10, 2, 2 * i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Image: {os.path.basename(image_path)}")

        plt.subplot(10,2, 2 * i + 2)
        plt.imshow(mask, cmap='gray')


    save_path = os.path.join(output_folder, f"plot_batch_{batch_index}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.tight_layout()
    plt.close()

    print(f"Saved plot {batch_index + 1}")


