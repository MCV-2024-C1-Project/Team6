import imageio
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

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