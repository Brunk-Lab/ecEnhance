#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2  
import numpy as np  
import math
import csv
import json
import pandas as pd  
from concurrent.futures import ProcessPoolExecutor, as_completed
from bayes_opt import BayesianOptimization  # Bayesian-Optimization 1.2.0
from bayes_opt.event import Events
import matplotlib
import matplotlib.pyplot as plt  
import seaborn as sns


# In[2]:


rgb_folder = r"/work/users/b/e/behnamie/FACS-FISH_redistribution_NCIH2170/version_2_Acc_87/2_optimization/input_directory/ground_truth_RGB"
dapi_folder = r"/work/users/b/e/behnamie/FACS-FISH_redistribution_NCIH2170/version_2_Acc_87/2_optimization/input_directory/ground_truth_DAPI"
ground_truth_csv = r"/work/users/b/e/behnamie/FACS-FISH_redistribution_NCIH2170/version_2_Acc_87/2_optimization/input_directory/GT.csv"
mia_predictions_csv = r"/work/users/b/e/behnamie/FACS-FISH_redistribution_NCIH2170/version_2_Acc_87/2_optimization/input_directory/output_mia.csv"  # Adjust path as needed

#Load ground truth data
ground_truth_df = pd.read_csv(ground_truth_csv).sample(n=300, random_state=10)


# In[3]:


def debug_save_image(image, name, step, out_folder, unique_id=""):
    """
    Saves intermediate images for debugging purposes.

    Parameters:
    image (numpy.ndarray): Image to save.
    name (str): Descriptive name of the image.
    step (int): Processing step number.
    out_folder (str): Output directory.
    unique_id (str): Unique identifier for the image.

    Returns:
    str: Filepath of the saved image.
    """
    filename = f"{unique_id}_{step:02d}_{name}.tif" if unique_id else f"{step:02d}_{name}.tif"
    filepath = os.path.join(out_folder, filename)
    # cv2.imwrite(filepath, image)  # Uncomment to save debug images
    print(f"Saved: {filepath}")
    return filepath

def extract_unique_id(filename, suffixes):
    """
    Extracts the base unique identifier from a filename by removing known suffixes.

    Parameters:
    filename (str): Name of the file.
    suffixes (list): List of possible suffixes to remove.

    Returns:
    str: Extracted unique identifier.
    """
    base = os.path.splitext(filename)[0]
    for suf in suffixes:
        if base.endswith(suf):
            return base[:-len(suf)]
    return base

def find_corresponding_dapi(unique_id, dapi_folder):
    """
    Locates the corresponding DAPI image for a given RGB image based on unique ID.

    Parameters:
    unique_id (str): Unique identifier of the image.
    dapi_folder (str): Directory containing DAPI images.

    Returns:
    str or None: Path to the DAPI image or None if not found.
    """
    dapi_suffixes = ["_DAPI", "_DAPI.tif", "_Merge.tif (RGB)", "_Merge.tif(RGB)",
                     "_Merge.tif (RGB).tif", "_Merge.tif(RGB).tif"]
    for fname in os.listdir(dapi_folder):
        if not fname.lower().endswith(('.tif', '.tiff', '.png')):
            continue
        uid = extract_unique_id(fname, dapi_suffixes)
        if uid == unique_id:
            return os.path.join(dapi_folder, fname)
    return None


# In[4]:


def mask_rgb_with_dapi(rgb_img, dapi_img):
    """
    Masks the RGB image using the DAPI image to focus on the nucleus.

    Parameters:
    rgb_img (numpy.ndarray): RGB image.
    dapi_img (numpy.ndarray): DAPI grayscale image.

    Returns:
    numpy.ndarray: Masked RGB image.

    Logic:
    Pixels where the DAPI image is black (intensity = 0) are set to black in the RGB image,
    isolating the region of interest (ROI) corresponding to the nucleus.
    """
    if len(dapi_img.shape) != 2:
        dapi_img = cv2.cvtColor(dapi_img, cv2.COLOR_BGR2GRAY)
    mask = (dapi_img == 0)
    rgb_masked = rgb_img.copy()
    rgb_masked[mask] = 0
    return rgb_masked

def top_hat_enhancement(gray, kernel_size=20, chrom_kernel_size=200, dampening_factor=0.6):
    """
    Enhances small bright features (ecDNAs) while suppressing large structures (chromosomes).

    Parameters:
    gray (numpy.ndarray): Grayscale image.
    kernel_size (int): Size of the structuring element for top-hat transformation.
    chrom_kernel_size (int): Size of the structuring element to estimate chromosomes.
    dampening_factor (float): Factor (0 < factor < 1) to dampen chromosome regions.

    Returns:
    numpy.ndarray: Enhanced image with suppressed chromosomes.

    Logic:

    Top-Hat Transformation: Morphological opening removes small features, and subtraction from the original image highlights ecDNAs.
    Chromosome Suppression: Morphological closing estimates large structures (chromosomes), 
    which are identified via Otsu thresholding and dampened to preserve nearby ecDNAs. """ 

    # Top-hat transformation
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(kernel_size), int(kernel_size)))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, se)
    top_hat = cv2.subtract(gray, opened)
    top_hat_norm = cv2.normalize(top_hat, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Estimate chromosomes
    chrom_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(chrom_kernel_size), int(chrom_kernel_size)))
    chromosome_est = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, chrom_se)
    _, chrom_mask = cv2.threshold(chromosome_est, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Dampen chromosome regions
    top_hat_soft = top_hat_norm.astype(np.float32)
    top_hat_soft[chrom_mask == 255] *= dampening_factor
    return np.clip(top_hat_soft, 0, 255).astype(np.uint8)
    

def custom_clahe(gray, clip_limit, tile_grid_size):
    """
    Applies Contrast Limited Adaptive Histogram Equalization to enhance local contrast.

    Parameters:
    gray (numpy.ndarray): Grayscale image.
    clip_limit (float): Threshold for contrast limiting.
    tile_grid_size (int or tuple): Size of the grid for histogram equalization.

    Returns:
    numpy.ndarray: Enhanced image.

    Logic:
    CLAHE adjusts intensity histograms locally, improving visibility of ecDNAs against varying backgrounds.
    """
    tile_size = (int(tile_grid_size), int(tile_grid_size))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(gray)

def apply_sharpening_gray(gray, strength):
    """
    Sharpens the image to enhance edges of ecDNAs.

    Parameters:
    gray (numpy.ndarray): Grayscale image.
    strength (float): Sharpening intensity.

    Returns:
    numpy.ndarray: Sharpened image.

    Logic:
    A high-pass filter amplifies edge gradients, making ecDNA boundaries more distinct.
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32) * (strength / 5.0)
    return cv2.filter2D(gray, -1, kernel)

def apply_sigmoid(gray, cutoff, gain):
    """
    Applies a sigmoid function to binarize bright spots.

    Parameters:
    gray (numpy.ndarray): Grayscale image.
    cutoff (float): Intensity threshold for sigmoid.
    gain (float): Steepness of the sigmoid curve.

    Returns:
    numpy.ndarray: Binarized image.
    
    Logic:
    Maps intensities to a binary-like output, emphasizing ecDNAs as bright spots.
    """
    norm = gray.astype(np.float32) / 255.0
    c = cutoff / 255.0
    out = 1.0 / (1.0 + np.exp(-gain * (norm - c)))
    return (out * 255).astype(np.uint8)


def process_images(input_image, params, save_debug=False, output_folder=None, unique_id=""):
    """
    Processes the input image through the enhancement pipeline.

    Parameters:
    input_image (numpy.ndarray): Input RGB or grayscale image.
    [See individual function docstrings for other parameters]

    Returns:
    tuple: (simple_gray, enhanced) - Original grayscale and enhanced images.
    
    Logic:
    Sequentially applies top-hat enhancement, CLAHE, sharpening, and sigmoid adjustment to isolate ecDNAs.
    """ 
    if input_image.ndim == 3:
        gray = cv2.cvtColor(cv2.convertScaleAbs(input_image), cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.convertScaleAbs(input_image)
    
    simple_gray = gray.copy()
    th = top_hat_enhancement(gray, params['kernel_size'], params['chrom_kernel_size'], params['dampening_factor'])
    sharp = apply_sharpening_gray(th, params['strength'])    
    clahe_img = custom_clahe(sharp, params['clip_limit'], params['tile_grid_size'])
    enhanced = apply_sigmoid(clahe_img, params['cutoff'], params['gain'])
    
    if save_debug and output_folder:
        debug_save_image(simple_gray, "simple_gray", 1, output_folder, unique_id)
        debug_save_image(enhanced, "enhanced_gray", 2, output_folder, unique_id)
    
    return simple_gray, enhanced


# In[5]:


def merge_close_objects(objects, merge_distance):
    """
    Merges objects closer than a specified distance to avoid over-counting.

    Parameters:
    objects (list): List of detected objects with 'bbox', 'centroid', and 'area'.
    merge_distance (float): Maximum distance to merge objects.
    
    Returns:
    list: Merged objects.

    Logic:
    Combines nearby objects, likely fragments of the same ecDNA, based on centroid proximity.
    """

    merged_objects = []
    taken = [False] * len(objects)
    for i in range(len(objects)):
        if taken[i]:
            continue
        current = objects[i].copy()
        for j in range(i + 1, len(objects)):
            if taken[j]:
                continue
            if math.dist(current["centroid"], objects[j]["centroid"]) < merge_distance:
                x1, y1, w1, h1 = current["bbox"]
                x2, y2, w2, h2 = objects[j]["bbox"]
                current["bbox"] = (min(x1, x2), min(y1, y2), max(x1 + w1, x2 + w2) - min(x1, x2), max(y1 + h1, y2 + h2) - min(y1, y2))
                cx1, cy1 = current["centroid"]
                cx2, cy2 = objects[j]["centroid"]
                current["centroid"] = ((cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0)
                current["area"] += objects[j]["area"]
                taken[j] = True
        merged_objects.append(current)
        taken[i] = True
    return merged_objects

def classify_as_white_or_ecDNA(roi, white_value_threshold=150, white_saturation_threshold=45):
    """
    Classifies objects as 'chromosome' (white) or 'ecDNA' based on HSV color.

    Parameters:
    roi (numpy.ndarray): Region of interest from RGB image.
    white_value_threshold (float): Minimum brightness (V) for white classification.
    white_saturation_threshold (float): Maximum saturation (S) for white classification.

    Returns:
    str: 'chromosome' or 'ecDNA'.

    Logic:
    In HSV space, white objects (chromosomes) have high brightness (V) and low saturation (S),
    while ecDNAs typically exhibit distinct colors.
    """

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _, S_mean, V_mean, _ = cv2.mean(hsv_roi)
    return "chromosome" if (V_mean > white_value_threshold and S_mean < white_saturation_threshold) else "ecDNA"

def object_detection_and_overlay(enhanced_img, rgb_img_path, params, output_folder, unique_id):
    """
    Detects objects in the enhanced image and classifies them using the RGB image.

    Parameters:
    enhanced_img (numpy.ndarray): Enhanced grayscale image.
    rgb_img_path (str): Path to the corresponding RGB image.
    [See individual function docstrings for other parameters]

    Returns:
    tuple: (total_count, merged_objects, counts) - Count of ecDNAs, detected objects, and classification counts.

    Logic:
    Uses connected components analysis to detect objects, merges close ones, and classifies them based on color.
    """
    _, thresh = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    objects = [{"bbox": (stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]), "area": stats[i, 4], "centroid": centroids[i]}
               for i in range(1, num_labels) if params['min_area'] <= stats[i, 4] <= params['max_area']]
    
    merged_objects = merge_close_objects(objects, params['merge_distance'])
    
    if os.path.exists(rgb_img_path):
        rgb_img = cv2.imread(rgb_img_path, cv2.IMREAD_COLOR)
        counts = {"chromosome": 0, "ecDNA": 0}
        for obj in merged_objects:
            x, y, w, h = obj["bbox"]
            roi = rgb_img[y:y+h, x:x+w]
            label = classify_as_white_or_ecDNA(roi, params['white_value_threshold'], params['white_saturation_threshold'])
            counts[label] += 1
    else:
        counts = {}
    
    return len(merged_objects), counts


# In[6]:


def median_absolute_percentage_error(mape_list):
    """
    Computes the median absolute percentage error (MdAPE).

    Parameters:
        mape_list (list): List of percentage errors.

    Returns:
        float: Median of the absolute percentage errors.

    Logic:
        MdAPE provides a robust central tendency measure, minimizing the impact of outliers in error distributions.
    """

    return np.median(mape_list) if mape_list else 1e6

def process_single_image(row, params):
    """
    Processes a single image and computes its MAPE.

    Parameters:
        row (pandas.Series): Ground truth row with 'unique_id' and 'ground_truth_count'.
        params (dict): Hyperparameters for the pipeline.

    Returns:
        float: MAPE for the image.

    Logic:
        Applies the full pipeline to an image and calculates the percentage error against ground truth.
    """
    try:
        unique_id = row["unique_id"]
        true_count = row["ground_truth_count"]
        if true_count == 0:
            return 1000  # High error for undefined MAPE

        rgb_path = os.path.join(rgb_folder, f"{unique_id}.tif")
        dapi_path = find_corresponding_dapi(unique_id, dapi_folder)

        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_img is None:
            return 1000

        if dapi_path:
            dapi_img = cv2.imread(dapi_path, cv2.IMREAD_GRAYSCALE)
            if dapi_img is not None:
                rgb_img = mask_rgb_with_dapi(rgb_img, dapi_img)

        _, enhanced = process_images(rgb_img, params)
        total_count, _ = object_detection_and_overlay(enhanced, rgb_path, params, "", unique_id)
        return 100.0 * abs(total_count - true_count) / true_count

    except Exception as e:
        # Log the error (or print) and return a high error value
        print(f"Error processing image {row['unique_id']}: {e}")
        return 1000

    


def objective_function(kernel_size, clip_limit, tile_grid_size, strength, cutoff, gain,
                      merge_distance, min_area, max_area, white_value_threshold,
                      white_saturation_threshold, chrom_kernel_size, dampening_factor):
    """
    Objective function for Bayesian Optimization to minimize MdAPE.

    Parameters:
        [See individual function docstrings for parameter details]

    Returns:
        float: Negative MdAPE (for maximization in Bayesian Optimization).

    Logic:
        Evaluates the pipeline across all images in parallel, computing MdAPE to guide parameter optimization.
    """
    params = {
        'kernel_size': kernel_size,
        'strength': strength,
        'clip_limit': clip_limit,
        'tile_grid_size': tile_grid_size,
        'cutoff': cutoff,
        'gain': gain,
        'chrom_kernel_size': chrom_kernel_size,
        'dampening_factor': dampening_factor,
        'merge_distance': merge_distance,
        'min_area': min_area,
        'max_area': max_area,
        'white_value_threshold': white_value_threshold,
        'white_saturation_threshold': white_saturation_threshold
    }
    
    ground_truth_df = pd.read_csv(ground_truth_csv)
    mape_values = []

    from concurrent.futures import ThreadPoolExecutor
    max_workers = os.cpu_count() 
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, row, params) for _, row in ground_truth_df.iterrows()]
        for future in as_completed(futures):
            mape_values.append(future.result())

    
    mdape = median_absolute_percentage_error(mape_values)
    return -mdape  # Negative for maximization


# In[7]:


if __name__ == '__main__':
    pbounds = {
        'kernel_size': (10, 50),
        'strength': (2.0, 9.5),
        'clip_limit': (0.1, 5.0),
        'tile_grid_size': (10, 50),
        'cutoff': (50, 150),
        'gain': (1, 50),
        'chrom_kernel_size': (100, 300),
        'dampening_factor': (0.1, 0.9),
        'merge_distance': (1, 20),
        'min_area': (1, 20),
        'max_area': (400, 1000),
        'white_value_threshold': (100, 200),
        'white_saturation_threshold': (20, 80)
    }
    
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=10,
        verbose=2
    )
    
    # Run optimization: 10 initial points for exploration, 50 iterations for refinement
    optimizer.maximize(init_points=15, n_iter=85)
    
    print("Best Parameters:", optimizer.max)
    
    # Save best parameters
    best_params = optimizer.max['params']
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f)

