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
        # SIFT_nfeatures_nOctaveLayers_contrastThreshold
        nfeatures = int(parts[1]) if len(parts) > 1 else 0  # 0 means no limit
        nOctaveLayers = int(parts[2]) if len(parts) > 2 else 3
        contrastThreshold = float(parts[3]) if len(parts) > 3 else 0.04
        return SIFT(nfeatures, nOctaveLayers, contrastThreshold)
    elif class_name == "ORB":
        if len(parts) > 1:
            nfeatures = int(parts[1])
            WTA_K = int(parts[2]) if len(parts) > 2 else 2
            fastThreshold = int(parts[3]) if len(parts) > 3 else 20
            return ORB(nfeatures, WTA_K, fastThreshold)
        else:
            return ORB()
    elif class_name == "PCASIFT":
        # PCASIFT_nComponents_response_octave
        # n_components = int(parts[1]) if len(parts) > 1 else 64
        # response_threshold = float(parts[2]) if len(parts) > 2 else 0.6
        # octave_threshold = int(parts[3]) if len(parts) > 3 else 2
        # return PCASIFT(n_components, response_threshold, octave_threshold)
        return None
    elif class_name == "KAZE":
        if len(parts) > 1:
            extended, threshold, nOctaves, nOctaveLayers = parts[1:]
            
            if(parts[1] == 1):extended=True
            else: extended= False
            threshold = float(parts[2]) if len(parts) > 2 else 0.001
            nOctaves = int(parts[3]) if len(parts) > 3 else 4
            nOctaveLayers = int(parts[4]) if len(parts) > 4 else 4
            return KAZE(extended, threshold, nOctaves, nOctaveLayers)
        else:
            return KAZE()
    else:
        sys.exit(f"ERROR: Unknown keypoint detector type '{type_str}'")



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
        
        threshold = self.threshold * harris_corners.max()
        keypoints = np.argwhere(harris_corners > threshold)
        
        return keypoints

# https://www.researchgate.net/publication/235355151_Scale_Invariant_Feature_Transform
class SIFT(KeypointAndDescriptorExtractor):
    def __init__(self, nfeatures, nOctaveLayers, contrastThreshold):
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold
        )

    def extract(self, image):
        image = resize_image(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.sift.detectAndCompute(gray_image, None)
        return descriptors
    
    
class ORB(KeypointAndDescriptorExtractor):
    def __init__(self, nfeatures=500, WTA_K=2, fastThreshold=20):
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,         # Start from the first pyramid level
            WTA_K=WTA_K,          # Use a larger patch size around keypoints
            fastThreshold=fastThreshold          # Lower threshold to capture more keypoints
        )
    def extract(self, image):
        image = resize_image(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)
        
        return descriptors

class KAZE(KeypointAndDescriptorExtractor):
    def __init__(self, extended=False, threshold=0.001, nOctaves=4, nOctaveLayers=4):
        self.kaze = cv2.KAZE_create(
            extended=extended,          # Default: False
            threshold=threshold,        # Default: 0.001
            nOctaves=nOctaves,          # Default: 4
            nOctaveLayers=nOctaveLayers # Default: 4
        )
    
    def extract(self, image):
        image = resize_image(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        keypoints, descriptors = self.kaze.detectAndCompute(gray_image, None)
        
        return descriptors

# https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=eb14821f5908e614a72ca1e66664582938643d51#:~:text=PCA%2Dbased%20SIFT%20descriptors&text=This%20feature%20vector%20is%20sig,same%20keypoint%20in%20different%20images.
class PCASIFT(KeypointAndDescriptorExtractor):
    def __init__(self, n_components=64, response_threshold=0.6, octave_threshold=2):
        self.n_components = n_components
        self.response_threshold = response_threshold
        self.octave_threshold = octave_threshold

    def extract(self, image):
        image = resize_image(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)

        if not keypoints or descriptors is None:
            return [], None

        max_response = max(kp.response for kp in keypoints)
        filtered_keypoints = [
            kp for kp in keypoints
            if kp.response >= self.response_threshold * max_response and kp.octave >= self.octave_threshold
        ]
        filtered_descriptors = descriptors[
            [i for i, kp in enumerate(keypoints) if kp in filtered_keypoints]
        ] if descriptors is not None else None

        if filtered_descriptors is not None:
            pca = PCA(n_components=self.n_components)
            reduced_descriptors = pca.fit_transform(filtered_descriptors)
        else:
            reduced_descriptors = None

        return reduced_descriptors

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
    # disp_img = resize_image(image)
    # disp_img = cv2.drawKeypoints(disp_img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(disp_img), plt.show()

    # Load the input image


    # pcasift = LocalFeatureExtractorFactory("PCASIFT_64")
    # keypoints, descriptors = pcasift.extract(image)
    # print(len(keypoints), descriptors.shape)


    # disp_img = resize_image(image)
    # disp_img = cv2.drawKeypoints(disp_img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(disp_img)
    # plt.title(f"Detected Keypoints: {len(keypoints)}")
    # plt.show()

