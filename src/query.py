import imageio
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import descriptors.compute_descriptor as compute_descriptor
import descriptor.get_all_jpg_images as get_all_jpg_images
import measures.MeasureFactory as MeasureFactory

def run_query_week1(input_path, output_path):
    pass

def single_prediction(input_image, output_path):
    image = imageio.imread(input_image)
    gray_image = rgb2gray(image)
    
    fig, ax = plt.subplots(1,1)

    ax.hist(gray_image.flatten())
    ax.set(title='r')

    plt.show()
    plt.close()

def retrieve_K(descriptor, db_descriptor, k):
    measure = MeasureFactory("HellingerKernel")
    # first interation can  been improved O(n²) depending of sort implementation
    result = [(path, measure(descriptor,db_image_descriptor)) for path, db_image_descriptor in db_descriptor]
    result = sorted(result, key=lambda (path, score): score) # we sort the list by using a function that extract the score
    return result[:k] # return the first k element of the list

def prediction(input_path, db_path, k, descriptor_type, descriptor_subtype, evaluate=False, single_image=False):
    db_descriptor = load_descriptors(db_path) # format: [(path1, descriptor1),(path2, descriptor2),...,(pathN, descriptorN)]
    image_paths = [input_path] if single else get_all_jpg_images(input_path)
    result = []

    for path in image_paths:
        descriptor = compute_descriptor(path, descriptor_type, descriptor_subtype)
        k_result = retrieve_K(descriptor, db_descriptor, k) # [(path_resulting_image, metric) ... ]
        result.append((path, k_result)) # [(path_query_image, (path_resulting_image, metric)) ... ]
    
    if evaluate:
        #TODO compute hte mAP@K
    else:
        return result