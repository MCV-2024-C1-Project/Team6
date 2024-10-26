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
import scipy as sp
from skimage import feature
from skimage.transform import probabilistic_hough_line
import skimage.draw as draw
from skimage.morphology import convex_hull_image
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from shapely.geometry import Polygon


def area_difference(quad_points, polygon_points):
    quad_points = np.reshape(quad_points, (4, 2))
    quadrilateral = Polygon(quad_points)
    polygon = Polygon(polygon_points)
    outside_area = polygon.union(quadrilateral).area - polygon.intersection(quadrilateral).area
    return outside_area

def smallest_fitting_quad(polygon_points):
    points = np.array(polygon_points, dtype=np.float32)
    rect = cv2.minAreaRect(points)
    initial_quad = cv2.boxPoints(rect)
    initial_quad = initial_quad.astype(np.int32)
    x0 = initial_quad.flatten()
    result = minimize(area_difference, x0, args=(polygon_points,), method='L-BFGS-B')
    best_quad = result.x.reshape(4, 2)
    return best_quad.tolist()


def sort_quad_points(quad_points):
    quad_points = np.array(quad_points, dtype=np.float32)
    center = np.mean(quad_points, axis=0)
    def angle_from_center(point):
        angle = np.arctan2(point[1] - center[1], point[0] - center[0])
        return angle
    sorted_points = sorted(quad_points, key=angle_from_center)
    top_points = sorted(sorted_points[:2], key=lambda x: x[0])
    bottom_points = sorted(sorted_points[2:], key=lambda x: x[0], reverse=True)
    return np.array([top_points[0], top_points[1], bottom_points[0], bottom_points[1]], dtype=np.float32)

def extract_and_rectify_region(image, quad_points):
    quad_points = sort_quad_points(quad_points)
    width_top = np.linalg.norm(quad_points[1] - quad_points[0])
    width_bottom = np.linalg.norm(quad_points[2] - quad_points[3])
    height_left = np.linalg.norm(quad_points[3] - quad_points[0])
    height_right = np.linalg.norm(quad_points[2] - quad_points[1])
    width = int(max(width_top, width_bottom))
    height = int(max(height_left, height_right))

    dest_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad_points, dest_points)
    rectified_image = cv2.warpPerspective(image, M, (width, height))

    return rectified_image

def reorder_labels(label_map):
    num_labels = label_map.max()
    centroids = sp.ndimage.center_of_mass(label_map > 0, label_map, range(1, num_labels + 1))
    label_centroids = [(i + 1, centroid) for i, centroid in enumerate(centroids)]
    label_centroids.sort(key=lambda item: (item[1][1], item[1][0]))
    label_mapping = {label: new_label for new_label, (label, _) in enumerate(label_centroids, start=1)}
    reordered_label_map = np.zeros_like(label_map)
    for old_label, new_label in label_mapping.items():
        reordered_label_map[label_map == old_label] = new_label
    return reordered_label_map

def denoise(image):
    result = np.zeros_like(image)
    for i in range(3):
        result[:,:,i] = sp.ndimage.median_filter(image[:,:,i], size=7)
    return result

# sigma=2, threshold=50
def edge_detection(image, sigma=1.5, threshold=55):
    edges = np.zeros_like(image)
    for i in range(3):
        edges[:,:,i] = feature.canny(image[:,:,i], sigma=sigma, high_threshold=threshold, low_threshold=threshold*0.5)
    result = np.max(edges, axis=2)>0
    return result

def frame_filling(image, edges, line_gap=0.01, threshold=0.1):
    segment_threshold = int(min(edges.shape[0], edges.shape[1]) * threshold)
    line_gap_it = int(min(edges.shape[0], edges.shape[1]) * line_gap)
    edges_d = sp.ndimage.binary_dilation(edges, iterations=line_gap_it)
    insides = sp.ndimage.binary_fill_holes(edges_d)
    insides = sp.ndimage.binary_erosion(insides, iterations=line_gap_it)
    insides = sp.ndimage.binary_opening(insides, iterations=line_gap_it+2)
    # _, ax = plt.subplots(1, 2)
    # ax[0].imshow(image)
    # ax[1].imshow(insides)
    # plt.show()
    insides = sp.ndimage.binary_propagation(sp.ndimage.binary_erosion(insides, iterations=segment_threshold), mask=insides)
    return insides

def frame_segmenter(image):
    edges = edge_detection(image)
    frames = frame_filling(image, edges)
    labeled_array, num_features = sp.ndimage.label(frames)
    labeled_array = reorder_labels(labeled_array)
    return labeled_array, num_features

def get_best_quad_fit(region):
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    hull = cv2.approxPolyDP(hull, epsilon, True)
    corners = hull.reshape(-1, 2)
    quad = smallest_fitting_quad(corners)
    return quad

def extract_frame(image, region):
    region2 = convex_hull_image(region)
    quad = get_best_quad_fit(region2)
    return extract_and_rectify_region(image, quad), region2

def frame_detector(image):
    image_f = denoise(image)
    segments, num_features = frame_segmenter(image_f)
    result_mask = np.zeros_like(segments)
    result_frames = []
    for i in range(num_features):
        frame, final_mask = extract_frame(image_f, (segments==(i+1)))
        result_mask |= final_mask
        result_frames.append(frame)
    return result_mask, result_frames

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
            print(i)
            image = imageio.imread(image_path)
            gt_mask = imageio.imread(image_path[:-4] + ".png")
            pred_mask, frames = frame_detector(image)
            # _, ax = plt.subplots(1, 2)
            # ax[0].imshow(image)
            # ax[1].imshow(pred_mask)
            # plt.show()
            metric = evaluate(gt_mask, pred_mask)
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
    dataset_folder = "data/week3/qsd2_w3"
    output_folder = "data/week3/plots"
    results_folder = "data/week3/results"
    test_background_removal(dataset_folder, results_folder, plot=True, save=False)