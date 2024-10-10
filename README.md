# PROJECT: IMAGE RETRIEVAL

## Description

This project implements an image retrieval system that uses many types of image descriptors (color, texture, keypoints, etc.) to find the most similar images in a museum dataset. For each image in the query dataset, the system retrieves the top K most similar images from the museum dataset, ordered by their similarity scores. 

* Week 1: Build retrieval system based on 1D histogram of color.
* Week 2:


## Table of Contents

* [Dependencies](#dependencies)
* [Installation](#installation)
* [Project Structure](#project-structure)
* [Usage](#usage)

## Dependencies

The project requires the following packages:

* Python 3.12
* NumPy
* OpenCV-Python
* scikit-learn
* matplotlib
* scikit-image
* pandas
* SciPy
* Pillow
* ImageIO

These can be installed using:

```
pip install -r requirements.txt
```

## Installation

To set up the project on your local machine:

1.	Clone the repository:

```
git clone https://github.com/MCV-2024-C1-Project/Team6.git
cd Team6
```

2.	Create a virtual environment (optional but recommended):

* Using venv:

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

* Or using [Anaconda](https://docs.anaconda.com/anaconda/install/):

```
conda create -n your_env_name python=3.12
conda activate your_env_name
```

3.	Install the required dependencies:

```
pip install -r requirements.txt
```

## Project Structure

```
Team6/
│ 
├── README.md
├── requirements.txt
├── main.py
│ 
├── generated_descriptors
│   ├── descriptor_histogram_hsv_128_bins.pkl
│   └── descriptor_histogram_super_64_bins.pkl
│ 
├── result
│   └── result.pkl
│ 
├── src
│   ├── descriptors.py       # Functions to generate descriptors
│   ├── histogram.py         # Histogram implementations
│   ├── measures.py          # Distance/similarity measures
│   ├── performance.py       # Metrics like MAP@K
│   ├── plotting.py          # Plotting functions
│   └── query.py             # Predict function to query from museum database
│ ...
│ <Unnecessary files>
```

* main.py: Main script orchestrating the tool’s functionality.
* generated_descriptors/: Contains precomputed image descriptors (e.g., .pkl files).
* result/: Stores prediction results (e.g., result.pkl).
* src/: Source code directory containing modules for descriptors, histograms, measures, performance metrics, plotting, and querying.

## Usage

The main script for this project is main.py, which provides different subcommands for various functionalities. Below are the general usage instructions and available options.

### General Command Structure

```
python main.py [command] [options]
```

### Available Commands

1.	`init`: Generate image descriptors for the database.
2.	`predict`: Retrieve the top K most similar images for a set of query images.

### Command Details

#### 1. Initialize Database Descriptors

**Generate image descriptors for the database images.**

```
python main.py init --db_path <path_to_database> --descriptor-type <descriptor_type>
```


**Options:**

* --db_path: Path to the database images. Default is BBDD/.
* --descriptor-type: The type of descriptor to use. Format: Histogram-<color_mode>-<histogram_bins>. Default is Histogram-RGB-64.

**Available Descriptor Types:**

* Histogram Color Mode:

    * RGB:	Red, Green, Blue
    * GRAY:	Gray
    * HSV:	Hue, Saturation, Value
    * YCbCr:	Luma, Cb, Cr
    * Super:	Luma, Cb, Cr, Hue, Saturation, Value


* Example Descriptor Types:

    * Histogram-RGB-64
    * Histogram-HSV-128
    * Histogram-GRAY-256

#### 2. Retrieve Top K Similar Images

Retrieve the top K most similar images for the input query images.

```
python main.py predict --input <path_to_query_images> --result-number K --descriptor-type <descriptor_type> --measure <measure> [options]
```

**Options:**

* --input: Path to the input image or folder. Default is qsd1_w1/.
* --result-number: Number of top results to return (K). Default is 1.
* --descriptor-type: Descriptor type to be used, in the format Histogram-<color_mode>-<histogram_bins>. Default is Histogram-Super-64.
* --measure: Measure function to be used for similarity ranking. Default is HellingerKernel-Median.
* --plot: If set, show result plots.
* --evaluate: If set, perform evaluation using ground truth.
* --save-output: If set, save the prediction results to a pickle file.
* --output: Directory to save output files if --save-output is set. Default is results/.

**Available Similarity Measures:**

* HellingerKernel
* HellingerKernel-Median
* Intersection
* Intersection-Median
* L1
* L1-Median
* L2
* X2 (Chi-Squared Distance)
* X2-Median
* LInfinity
* KLD-Median
