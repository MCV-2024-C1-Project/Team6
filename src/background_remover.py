import imageio
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2


#background removal function
def remove_background(image,depth, K, sample_frequency,kernel):
    
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

    #calculate mean and std for each channel to use for image segemntation
    mean_color = [np.mean(R_samples), np.mean(G_samples), np.mean(B_samples)]
    std_color = [np.std(R_samples), np.std(G_samples), np.std(B_samples)]

    mask = ((image[:,:,0] > mean_color[0] - K * std_color[0]) & (image[:,:,0] < mean_color[0] + K * std_color[0]) &
        (image[:,:,1] > mean_color[1] - K * std_color[1]) & (image[:,:,1] < mean_color[1] + K * std_color[1]) &
        (image[:,:,2] > mean_color[2] - K * std_color[2]) & (image[:,:,2] < mean_color[2] + K * std_color[2]))

    mask = np.logical_not(mask).astype('uint8')
    
    #all images have at least 10 px of background
    mask[:10, :] = 0
    mask[-10:, :] = 0
    mask[:, :10] = 0  
    mask[:, -10:] = 0

    #opening the mask to eliminate noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    #generating bounding box that encapsules all values > 0 in the mask
    start_point, end_point = get_img_bbox(mask)

    #generating mask from bbox
    pred_mask = np.zeros(shape=(image.shape[0],image.shape[1]))
    pred_mask[start_point[1]:end_point[1], start_point[0]:end_point[0]] = 1

    
    return pred_mask


#bounding box retrieval function for a binary mask
def get_img_bbox(img):
    non_zero_indices = np.argwhere(img)
    if len(non_zero_indices) == 0:
        return tuple((0, size, size) for size in img.shape)

    min_indices = np.min(non_zero_indices,axis=0)
    max_indices = np.max(non_zero_indices,axis=0)

    bounding_box_min = tuple(min_indices[::-1])
    bounding_box_max = tuple(max_indices[::-1])

    return bounding_box_min,bounding_box_max


#function that calculates precision, recall and f1 for a pair of gt and pred mask
def evaluate(gt_mask,pred_mask,verbose = False):
    gt_mask_binary = (gt_mask > 0).astype('uint8')
    predicted_mask_binary = (pred_mask > 0).astype('uint8')

    tp = np.sum(np.bitwise_and(gt_mask_binary,predicted_mask_binary))
    fp = np.sum(np.bitwise_and(np.invert(gt_mask_binary),predicted_mask_binary))
    fn = np.sum(np.bitwise_and(gt_mask_binary,np.invert(predicted_mask_binary)))

    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * prec * recall) / (prec + recall)

    if(verbose):
        print(f'tp: {tp}, fp: {fp}, fn: {fn}')
        print(f'prec: {prec}, recall: {recall}, f1: {f1}')
    
    return prec,recall,f1


#function for quantitative testing of background removal
#dataset requires original images and masks in format name.jpg and name.png, respectively
def test_background_removal(input_folder,output_folder,plot = False, depth = 8, K = 2.5, sample_frequency = 50,kernel = np.ones((5, 5), np.uint8)):
    images_list = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith(".jpg")]
    batch_size = 10
    num_batches = int(np.ceil(len(images_list) / batch_size))
    metrics = []

    for batch_index in range(num_batches):
        
        if plot: plt.figure(figsize=(6, 20)) 
        batch_images = images_list[batch_index * batch_size:(batch_index + 1) * batch_size]

        for i, image_path in enumerate(batch_images):

            image = imageio.imread(image_path)
            gt_mask = imageio.imread(image_path[:-4] + ".png")
            pred_mask = remove_background(image, depth, K, sample_frequency,kernel)
            metric = evaluate(gt_mask,pred_mask)
            metrics.append(metric)
            if(plot):
                plt.subplot(10, 2, 2 * i + 1)
                plt.imshow(image)
                plt.axis('off')
                plt.title(f'prec: {metric[0]}, recall: {metric[1]}, f1: {metric[2]}')

                plt.subplot(10,2, 2 * i + 2)
                plt.axis('off')
                plt.imshow(pred_mask, cmap='gray')
                


        save_path = os.path.join(output_folder, f"plot_batch_{batch_index}.png")
        if plot:
            plt.savefig(save_path, bbox_inches='tight')
            plt.tight_layout()
            plt.close()
            print(f"Saved plot {batch_index + 1}")

    metrics = np.array(metrics)

    prec,recall,f1 = np.round(np.mean(metrics[:,0]),3),np.round(np.mean(metrics[:,1]),3),np.round(np.mean(metrics[:,2]),3)
    print(f'prec: {prec}, recall: {recall}, f1: {f1}')

if __name__ == '__main__':
    dataset_folder = "target/qsd2_w1"
    output_folder = "target/plots"
    test_background_removal(dataset_folder,output_folder,plot=False)