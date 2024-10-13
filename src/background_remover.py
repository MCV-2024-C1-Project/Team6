import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
import scipy.ndimage as nd
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import math

def plot_demo(image, mask, title):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image)
    ax[0].set_title("Image")
    ax[0].axis('off')
    mask_image = np.zeros(shape=(*mask.shape, 4))
    mask_image[:,:,0] = 1
    mask_image[:,:,1] = 0
    mask_image[:,:,2] = 0
    mask_image[:,:,3] = mask*0.5
    ax[1].imshow(image)
    ax[1].imshow(mask_image)
    ax[1].set_title(title)
    ax[1].axis('off')
    plt.show()

def plot_demo2(image1, image2, title1, title2):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image1)
    ax[0].set_title(title1)
    ax[0].set_title(title1)
    ax[0].axis('off')
    ax[1].imshow(image2)
    ax[1].set_title(title2)
    ax[1].axis('off')
    plt.show()

def quadrilater_fitting(foreground):
    def edge_distance(e0, e1, p):
        v = e1 - e0
        w = p - e0
        t = np.dot(w, v) / np.dot(v, v)
        t = max(0, min(1, t))
        closest_point = e0 + t * v
        distance = np.linalg.norm(p - closest_point)
        return distance

    def objective_function(quad_vertices, boundary_points):
        quad_vertices = quad_vertices.reshape(4, 2)
        distances = []
        for pt in boundary_points:
            dist0 = edge_distance(quad_vertices[0], quad_vertices[1], pt)
            dist1 = edge_distance(quad_vertices[1], quad_vertices[2], pt)
            dist2 = edge_distance(quad_vertices[2], quad_vertices[3], pt)
            dist3 = edge_distance(quad_vertices[3], quad_vertices[0], pt)
            distances.append(min(min(dist0, dist1), min(dist2, dist3)))
        return np.sum(distances)
    
    contour = foreground & ~nd.binary_erosion(foreground)
    points = np.argwhere(contour)
    points2 = points[np.linspace(0, points.shape[0]-1, 100, dtype=int)]
    (minx, miny), (maxx, maxy) = get_img_bbox(contour)
    
    initial_guess = np.array([[miny, minx], [maxy, minx], [maxy, maxx], [miny, maxx]])
    bounds = [
        (miny, None),
        (minx, None),
        (None, maxy),
        (minx, None),
        (None, maxy),
        (None, maxx),
        (miny, None),
        (None, maxx),
    ]

    result = minimize(lambda r: objective_function(r, points2), initial_guess.reshape(-1), options={'maxiter': 100}, bounds=bounds)
    quad_vertices = result.x.reshape(4, 2)
    quad_vertices_int = quad_vertices.astype(np.int32)
    blank_image = np.zeros_like(foreground, dtype=np.uint8)
    cv2.fillPoly(blank_image, [quad_vertices_int[:, [1,0]]], 1)
    return blank_image

def background_color_sampling1(image, radius):
    h, w, ch = image.shape
    north_colors = image[0:radius, :, :]
    south_colors = image[h-radius:h, :, :]
    west_colors = image[radius:h-radius, 0:radius, :]
    east_colors = image[radius:h-radius, w-radius:w, :]
    border_colors = np.concatenate([north_colors.reshape(-1, 3), 
                                    south_colors.reshape(-1, 3),
                                    west_colors.reshape(-1, 3),
                                    east_colors.reshape(-1, 3)])
    return np.mean(border_colors, axis=0), np.std(border_colors, axis=0)

def background_color_sampling2(image, radius, sampling_sides=None):
    h, w, ch = image.shape
    colors = []
    if sampling_sides is None:
        sampling_sides = ['north', 'south', 'west', 'east']
    if 'north' in sampling_sides:
        colors.append(image[0:radius, :, :].reshape(-1, 3))
    if 'south' in sampling_sides:
        colors.append(image[h-radius:h, :, :].reshape(-1, 3))
    if 'west' in sampling_sides:
        colors.append(image[radius:h-radius, 0:radius, :].reshape(-1, 3))
    if 'east' in sampling_sides:
        colors.append(image[radius:h-radius, w-radius:w, :].reshape(-1, 3))
    border_colors = np.concatenate(colors)
    pca = PCA(n_components=3)
    pca.fit(border_colors)
    return pca

def foreground_treshold1(input_image, threshold=2.5):
    h, w, ch = input_image.shape
    image = np.copy(input_image).astype(np.float32)
    mean, std = background_color_sampling1(image, 10)
    z_scores = np.abs(image.reshape(-1, 3) - mean) / std
    z_image = np.linalg.norm(z_scores, axis=1)
    z_image = z_image.reshape(h, w)
    foreground = z_image >= threshold
    return foreground

def foreground_treshold2(input_image, threshold=3.25):
    h, w, ch = input_image.shape
    image = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV).astype(np.float32)
    weights = np.array([0.5, 0.5, 2])
    mean, std = background_color_sampling1(image, 10)
    z_scores = np.abs(image.reshape(-1, 3) - mean) * weights / std
    z_image = np.linalg.norm(z_scores, axis=1)
    z_image = z_image.reshape(h, w)
    foreground = z_image >= threshold
    return foreground

def foreground_treshold3(input_image, threshold=4, sampling_sides=None):
    h, w, ch = input_image.shape
    border_pca = background_color_sampling2(input_image, 10, sampling_sides)
    explained_variance = np.sqrt(border_pca.explained_variance_)
    projected_data = border_pca.transform(input_image.reshape(-1, 3))
    z_scores = np.abs(projected_data) / explained_variance
    z_image = np.linalg.norm(z_scores, axis=1)
    z_image = z_image.reshape(h, w)
    foreground = z_image >= threshold
    return foreground

def foreground_treshold4(input_image, threshold=4):
    h, w = input_image.shape[:2]
    foreground = np.zeros((h, w), dtype=np.uint8)
    q1 = (slice(0, h//2), slice(0, w//2))
    q2 = (slice(0, h//2), slice(w//2, w))
    q3 = (slice(h//2, h), slice(0, w//2))
    q4 = (slice(h//2, h), slice(w//2, w))
    foreground[q1] = foreground_treshold3(input_image[*q1, :], threshold, sampling_sides=['north', 'west'])
    foreground[q2] = foreground_treshold3(input_image[*q2, :], threshold, sampling_sides=['north', 'east'])
    foreground[q3] = foreground_treshold3(input_image[*q3, :], threshold, sampling_sides=['south', 'west'])
    foreground[q4] = foreground_treshold3(input_image[*q4, :], threshold, sampling_sides=['south', 'east'])
    return foreground

def remove_mask_border(mask, radius):
    h, w = mask.shape
    mask = np.copy(mask)
    mask[0:radius, :] = 0
    mask[h-radius:h, :] = 0
    mask[radius:h-radius, 0:radius] = 0
    mask[radius:h-radius, w-radius:w] = 0
    return mask

def remove_holes(input, structure_size):
    mask = input>0
    mask_d = nd.binary_dilation(mask, iterations=structure_size)
    mask_d = nd.binary_fill_holes(mask_d)
    return nd.binary_erosion(mask_d, iterations=structure_size)

def get_largest_component(input, structure_size):
    mask = input>0
    mask_e = nd.binary_erosion(mask, iterations=structure_size)
    labeled_image, num_features = nd.label(mask_e)
    component_sizes = np.bincount(labeled_image.ravel())
    largest_component = component_sizes[1:].argmax() + 1
    largest_component_mask = labeled_image == largest_component
    result = nd.binary_dilation(largest_component_mask, iterations=structure_size)
    return result

def foreground_filter(foreground):
    foreground = remove_mask_border(foreground, 10)
    foreground = remove_holes(foreground, 5)
    foreground = get_largest_component(foreground, 5)
    foreground = nd.binary_opening(foreground, iterations=10)
    return foreground


def frame_detector(image):
    foreground = foreground_treshold3(image, 3)
    foreground = foreground_filter(foreground)
    # plot_demo2(foreground, foreground2, 'Mask Input', 'Mask Result')
    # plot_demo(image, foreground, 'Final Result')
    return foreground

def get_bbox(img):
    non_zero_indices = np.argwhere(img)
    if len(non_zero_indices) == 0:
        return tuple((0, size, size) for size in img.shape)

    min_indices = np.min(non_zero_indices,axis=0)
    max_indices = np.max(non_zero_indices,axis=0)

    return slice(min_indices[0], max_indices[0]), slice(min_indices[1], max_indices[1])

def crop_foreground(image, plot_debug=False):
    foreground = frame_detector(image)
    opening_size = 5
    foreground_e = nd.binary_erosion(foreground, iterations=opening_size)
    if plot_debug:
       pass
    bbox = get_bbox(foreground_e)
    cropped_image = np.copy(image[bbox])
    return cropped_image


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
def test_background_removal(input_folder,output_folder,plot = False, save=False, depth = 8, K = 2.5, sample_frequency = 50,kernel = np.ones((5, 5), np.uint8)):
    images_list = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith(".jpg")]
    batch_size = 10
    num_batches = int(np.ceil(len(images_list) / batch_size))
    metrics = []

    for batch_index in range(num_batches):
        
        if plot: plt.figure(figsize=(6, 20)) 
        batch_images = images_list[batch_index * batch_size:(batch_index + 1) * batch_size]

        for i, image_path in enumerate(batch_images):

            image = imageio.imread(image_path)
            gt_mask = imageio.imread(image_path[:-4] + ".jpg")
            
            pred_mask = remove_background(image, depth, K, sample_frequency,kernel)
            pred_mask = foreground_treshold1(image, 2.8)
            pred_mask = frame_detector(image)
            metric = evaluate(gt_mask,pred_mask)
            metrics.append(metric)
            if(plot):
                plt.subplot(10, 2, 2 * i + 1)
                plt.imshow(image)
                plt.axis('off')
                plt.title(f'prec: {metric[0]}, recall: {metric[1]}, f1: {metric[2]}')

                plt.subplot(10,2, 2 * i + 2)
                plt.axis('off')
                img_mask = np.zeros(shape=(*pred_mask.shape, 4))
                img_mask[:,:,3] = pred_mask*0.5
                img_mask[:,:,0] = 1
                plt.imshow(image)
                plt.imshow(img_mask)
            if(save):
                print(image_path)
                im_name = image_path.split('\\')[-1][:-4]+'.png'
                print(im_name)
                plt.imsave(os.path.join(output_folder, im_name), pred_mask, cmap='gray')
                pass
                


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
    dataset_folder = "data/week2/qst2_w2"
    output_folder = "data/week2/plots"
    results_folder = "data/week2/results"
    test_background_removal(dataset_folder, results_folder, plot=False, save=True)