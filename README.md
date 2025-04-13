# ecDNA Detection Pipeline

## Overview

This repository contains an automated pipeline for detecting and quantifying extrachromosomal DNA (ecDNA) in Fluorescence in situ Hybridization (FISH) images. The pipeline processes paired RGB FISH images and DAPI grayscale images to identify and classify ecDNA and chromosome-associated objects. It aims to improve upon manual counting and existing tools like MIA by offering a scalable, accurate, and automated solution.

### Main Components

The pipeline consists of the following key stages:

1. **Image Preprocessing**: Masks the RGB FISH image using the DAPI grayscale image to isolate nuclear regions, converting the result to grayscale for further processing.  
2. **Image Enhancement**: Applies a sequence of techniques to enhance ecDNA visibility:  
   - Top-Hat Filtering: Highlights small bright spots (ecDNA).  
   - Sharpening: Enhances object edges.  
   - CLAHE (Contrast Limited Adaptive Histogram Equalization): Improves contrast.  
   - Sigmoid Transformation: Boosts ecDNA signal clarity.  
3. **Object Detection**: Identifies potential ecDNA and chromosome objects through:  
   - Thresholding: Creates a binary image.  
   - Morphological Cleaning: Removes noise and refines objects.  
   - Connected Components Analysis: Labels objects with bounding boxes.  
   - Object Merging: Combines nearby objects.  
4. **Object Classification**: Classifies detected objects as 'ecDNA' or 'chromosomes' based on HSV color properties.  
5. **Hyperparameter Optimization**: Uses Bayesian Optimization to tune parameters, minimizing error (e.g., MdAPE).  
6. **Validation and Comparison**: Compares pipeline results to manual ground truth and MIA predictions using statistical metrics and visualizations.  
7. **Batch Processing**: Processes multiple images in parallel, generating structured outputs like CSV and JSON files.

---

## Setup Instructions

### **1. Clone the Repository**

#### Clone this repository to your local machine or server:

```bash
git clone https://github.com/your-repo/ecDNA-detection-pipeline.git
cd ecDNA-detection-pipeline
```

### **2. Create and Activate a Virtual Environment**
#### Use a virtual environment to manage dependencies:

```bash
python3 -m venv ecDNA_env
source ecDNA_env/bin/activate  # On Windows: ecDNA_env\Scripts\activate
```

### **3. Install Dependencies**
#### Install required Python packages from requirements.txt:
```bash
pip install -r requirements.txt
```

### Key dependencies include:
- opencv-python==4.11.0.86
- numpy==1.26.4
- pandas==2.1.4
- matplotlib==3.8.4
- seaborn==0.13.2
- scipy==1.13.1
- bayesian-optimization==2.0.3


### **4. Prepare Input Data**
#### Organize your input data as follows:

- **RGB Images:** Place in a directory (e.g., data/FISH_images/).
- **DAPI Images:** Place in a separate directory (e.g., data/DAPI_images/).
- **Ground Truth CSV (optional, for validation):** A CSV with columns unique_id and ground_truth_count.
- **MIA Predictions CSV (optional, for comparison):** A CSV with columns Image Name and Count.
- Update the paths in the configuration section of the notebooks or scripts to match your data locations.

## **Running the Pipeline**

### **1. Process a Single Image (Tutorial)**
#### Explore the pipeline step-by-step:
```Bash
jupyter notebook notebooks/01_ecDNA_counting_pipeline_tutorial.ipynb
```
#### This processes one image and visualizes each stage.

### **2. Hyperparameter Optimization (Optional)**
```bash
jupyter notebook notebooks/02_parameter_optimization.ipynb
```
#### This generates best_params.json with optimized settings.

### **3. Validation and Comparison**
#### Validate results against ground truth and MIA:
```bash
jupyter notebook notebooks/03_validation_and_comparison.ipynb
```
#### This generates performance metrics and comparison plots.


### **4. Batch Processing**
#### Process a large dataset:
```bash
jupyter notebook notebooks/04_batch_processing.ipynb
```
#### Alternatively, use the script:
```bash
python scripts/batch_processing.py --rgb_folder data/FISH_images/ --dapi_folder data/DAPI_images/ --output_folder results/
```

#### This processes all images and saves outputs in results/.

## **Interpreting the Outputs**

**1. Metrics Summary**
- metrics_summary.csv: Lists total object count, chromosome count, and ecDNA count per image.
updated_ecDNA_counts.csv: Adds comparison metrics (e.g., TP, FP, FN) with MIA.
**2. Object Details**
- ref_objects.json: Contains detailed object data (e.g., bounding boxes, centroids, areas) for each image.
**3. Debug Images**
- Intermediate images (e.g., masked RGB, enhanced grayscale, annotated overlays) are saved in subfolders (e.g., results/G0/High_HER2/) for inspection.
**4. Visualizations**
- Scatter plots and box plots comparing ecDNA counts to ground truth and MIA are saved in the output directory.

### Examples of Commands
#### Activate the Environment
```Bash
source ecDNA_env/bin/activate  # On Windows: ecDNA_env\Scripts\activate
```

### Additional Notes

- Reproducibility: A random seed of 10 is set for NumPy to ensure consistent results.
- Customization: Edit best_params.json or rerun optimization to adjust parameters.
- Troubleshooting: Ensure dependencies match requirements.txt to avoid version issues.
