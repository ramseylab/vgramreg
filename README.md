# vgramreg
Exploration of regression models for processing square-wave voltammograms to estimate analyte concentration.

This repository contains all the necessary code to replicate the results in the paper titled **"Evaluation of multi-feature machine-learning models for
analyzing electrochemical signals for drug monitoring"**

## Prerequisites
-   Python 3.8 or later
-   Anaconda or Miniconda

## Installation

### 1.  Create a Conda Environent
```bash 
conda create -n vgramreg python=3.9
conda activate vgramreg
```
### 2.  Install Dependencies
```bash
pip install -r requirements.txt
```

## Download Dataset
```bash
    wget ...
```
Extract the file and store the dataset in the root directory
```bash
.
├── ML1_ML2
│   ├── 2024_02_19_ML1
│   └── 2024_02_22_ML2
├── README.md
├── main.py
├── requirements.txt
└── src
```


## Generate Dataset
Open Config file src/config.py
```bash
    DATASET_PATH = '<full path locationo to ML1_ML2 dataset>'
    DATASET_PATH = '/Users/abc/Desktop/Epilepsey/Code/vgramreg/ML1_ML2'
```
```bash
    python src/generate_dataset.py
```
This code creates three.xlsx files in the respective folders (2024_02_19_ML1 and 2024_02_22_ML2). We will use only one file named extracted_features.xlsx.

```bash
ML1_ML2
├── 2024_02_19_ML1
│   ├── dataframe_log_NOrecenter_0.006_0_1.04_0.15_0.17extra_features.xlsx
│   ├── extracted_features.xlsx
│   └── stats_log_NOrecenter_0.006_0_1.04_0.15_0.17extra_features.xlsx
|
└──2024_02_22_ML2
    ├── dataframe_log_NOrecenter_0.006_0_1.04_0.15_0.17extra_features.xlsx
    ├── extracted_features.xlsx
    └── stats_log_NOrecenter_0.006_0_1.04_0.15_0.17extra_features.xlsx
```

## Running the models
```bash
python main.py
```

## Project Overview

All the graphs are stored in the folder named **Output**.

## Contents of the Output Folder

### 1. **Graphs**:
   - Contains images in `.png` format.
   - These images are bar charts comparing different models.

### 2. **Feature Selection List**:
   - A subfolder containing `.xlsx` files.
   - Each file has performance scores for each step in the feature selection process for each model.