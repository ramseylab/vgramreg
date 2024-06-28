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
conda create -n vgramreg python=3.8
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

## Generate Dataset
Open Config file src/config.py
```bash
    DATASET_PATH = '<path to ML1_ML2 dataset>'
    DATASET_PATH = 'ML1_ML2'
```
```bash
    python src/generate_dataset.py
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