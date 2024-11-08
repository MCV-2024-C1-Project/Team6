import cv2
from matplotlib import pyplot as plt
import numpy as np
import imageio.v3 as iio
import sys
from typing import Tuple, List

from sklearn.decomposition import PCA

def LocalFeatureExtractorFactory(type_str: str):
    parts = type_str.split('_')
    class_name = parts[0]
    
    if class_name == "SIFT":
        #no parameters
        return SIFT()
    elif class_name == "ORB":
        return ORB()
    elif class_name == "PCASIFT":
        # PCASIFT_nComponents
        n_components = int(parts[1]) if len(parts) > 1 else 64  # Default to 64 components
        return PCASIFT(n_components)
    else:
        sys.exit(f"ERROR: Unknown keypoint detector type '{type_str}'")
    # class_name == "HarrisCorner":
    #     # HarrisCornerDetector_blockSize_ksize_k_threshold
    #     block_size = int(parts[1]) if len(parts) > 1 else 2
    #     ksize = int(parts[2]) if len(parts) > 2 else 3
    #     k = float(parts[3]) if len(parts) > 3 else 0.04
    #     threshold = float(parts[4]) if len(parts) > 4 else 0.01
    #     return HarrisCornerDetector(block_size, ksize, k, threshold)



class KeypointDetector(object):
    def detect_keypoints(self, image) -> List:
        """
        Detects keypoints for the given image.

        Args:
            image: The input image to process.

        Returns:
            keypoints (List): A list of detected keypoints.
        """
        raise NotImplementedError("ERROR: detect_keypoints should be implemented by a subclass")
    
class LocalDescriptor(object):
    def compute_descriptor(self, image, keypoints) -> List:
        """
        Detects keypoints for the given image.

        Args:
            image: The input image to process.
            keypoints: A list of detected keypoints.

        Returns:
            descriptors (List): A list of corresponding descriptors.
        """
        raise NotImplementedError("ERROR: detect_keypoints should be implemented by a subclass")
class KeypointAndDescriptorExtractor(object):
    def extract(self, image) -> Tuple[List, List]:
        """
        Detects keypoints and computes descriptors for the given image.

        Args:
            image: The input image to process.

        Returns:
            Tuple[List, List]: A tuple containing two lists:
                - keypoints (List): A list of detected keypoints.
                - descriptors (List): A list of corresponding descriptors.
        """
        raise NotImplementedError("ERROR: detect_keypoints should be implemented by a subclass")

# blockSize - It is the size of neighbourhood considered for corner detection
# ksize - Aperture parameter of the Sobel derivative used.
# k - Harris detector free parameter in the equation.
# treshold the % of top corners
# https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
class HarrisCornerDetector(KeypointDetector):
    def __init__(self, block_size=2, ksize=3, k=0.04, threshold = 0.01):
        self.block_size = block_size
        self.ksize = ksize
        self.k = k
        self.threshold = threshold

    def detect_keypoints(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        harris_corners = cv2.cornerHarris(gray_image, blockSize=self.block_size, ksize=self.ksize, k=self.k)
        # harris_corners = cv2.dilate(harris_corners, None)
        
        threshold = self.threshold * harris_corners.max()
        keypoints = np.argwhere(harris_corners > threshold)
        
        return keypoints

# https://www.researchgate.net/publication/235355151_Scale_Invariant_Feature_Transform
class SIFT(KeypointAndDescriptorExtractor):
    def extract(self, image):
        image = resize_image(image, target_height=500)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        sift = cv2.SIFT_create()

        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        # keypoint_coords = [(int(kp.pt[1]), int(kp.pt[0])) for kp in keypoints]
        
        return keypoints, descriptors
    
    
class ORB(KeypointAndDescriptorExtractor):
    def extract(self, image):
        image = resize_image(image, target_height=512)
        print(image.shape)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        orb = cv2.ORB_create()

        keypoints, descriptors = orb.detectAndCompute(gray_image, None)
        # keypoint_coords = [(int(kp.pt[1]), int(kp.pt[0])) for kp in keypoints]
        
        return keypoints, descriptors

# https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=eb14821f5908e614a72ca1e66664582938643d51#:~:text=PCA%2Dbased%20SIFT%20descriptors&text=This%20feature%20vector%20is%20sig,same%20keypoint%20in%20different%20images.
class PCASIFT(KeypointAndDescriptorExtractor):
    def __init__(self, n_components=64):
        self.n_components = n_components

    def extract(self, image):
        image = resize_image(image, target_height=500)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)

        if descriptors is not None:
            # Apply PCA to reduce the descriptor length
            pca = PCA(n_components=self.n_components)
            reduced_descriptors = pca.fit_transform(descriptors)
        else:
            reduced_descriptors = None

        return keypoints, reduced_descriptors

def resize_image(image, target_height=512):
    height, width = image.shape[:2]
    scaling_factor = target_height / float(height)
    new_size = (int(width * scaling_factor), target_height)
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image


if __name__ == "__main__":
    # Load the input image
    # image = iio.imread('../W4/qsd1_w4/00025.jpg')
    image = iio.imread('target/qsd1_w1/00027.jpg')
    
    
    # # Choose the detector
    # locafeat = LocalFeatureExtractorFactory("ORB")
    # keypoints, descriptors = locafeat.extract(image)
    # # print(keypoints)
    # disp_img = resize_image(image, target_height=512)
    # disp_img = cv2.drawKeypoints(disp_img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(disp_img), plt.show()

    # Load the input image


    pcasift = LocalFeatureExtractorFactory("PCASIFT_64")
    keypoints, descriptors = pcasift.extract(image)
    print(len(keypoints), descriptors.shape)


    # Visualize the keypoints
    disp_img = resize_image(image, target_height=512)
    disp_img = cv2.drawKeypoints(disp_img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(disp_img)
    plt.title(f"Detected Keypoints: {len(keypoints)}")
    plt.show()

    # Detect keypoints
    # if isinstance(locafeat, SIFT):
    #     keypoints, descriptors = locafeat.extract(image)
    #     print(f"Number of SIFT keypoints detected: {len(keypoints)}")
    #     print(f"SIFT Descriptors shape: {descriptors.shape}")
    # else:
    #     keypoints = detector.detect_keypoints(image)
    #     descriptors = None
    #     print(f"Number of Harris keypoints detected: {len(keypoints)}")

    # # Visualization
    # dot_size = 2
    # image_copy = image.copy()
    # for (x, y) in keypoints:
    #     cv2.circle(image_copy, (y, x), dot_size, (0, 0, 255), dot_size)
    
    # plt.subplot(1, 2, 1)
    # plt.title("Original Image (Resized)")
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.subplot(1, 2, 2)
    # plt.title(f"Detected Keypoints (Count: {len(keypoints)})")
    # plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
    # plt.show()
