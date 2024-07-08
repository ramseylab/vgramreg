# vgramreg
Exploration of regression models for processing square-wave voltammograms to estimate analyte concentration.

This repository contains all the necessary code to replicate the results in the paper titled **"Evaluation of multi-feature machine-learning models for
analyzing electrochemical signals for drug monitoring"**

## Prerequisites
-   Python 3.8 or later
-   Anaconda or Miniconda (optional, can use a python virtualenv instead)

## Installation

### 1.  Create a Python3.9 Environent

This can be done using Conda or a python virtualenv.

#### Using Conda:
```bash 
conda create -n vgramreg python=3.9
conda activate vgramreg
```

#### Using a python virtualenv:
```bash
python3.9 -m venv venv
source venv/bin/activate
```

### 2.  Install Dependencies
```bash
pip install -r requirements.txt
```

## Download Dataset
```bash
    wget ...
```
Extract the file and store the dataset in the `vgramreg` project root directory as shown below:
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
In a text editor, open the connfiguration file `src/config.py`, and change
`DATASET_PATH` so that it is set to the full absolute path location to the `ML1_ML2` 
folder, e.g.:
```bash
    DATASET_PATH = '/Users/abc/Desktop/Epilepsey/Code/vgramreg/ML1_ML2'
```
Save and exit your editor. Then, in a bash session where your current working
directory is in the `vgramreg` project root directory, generate Excel spreadsheets of the raw voltammogram files 
by running the `generate_dataset.py` script:
```bash
    python src/generate_dataset.py
```
This code creates three `.xlsx` files in the respective folders (2024_02_19_ML1 and 2024_02_22_ML2). We will use only one file named `extracted_features.xlsx`.

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
The code should generate (up to small differences in p-values due to random sampling), 
the following text output on `stdout`:
```
(venv) sramsey-laptop:vgramreg sramsey$ python main.py
######Data Distribution:#########
Training {0: 50, 16: 51, 8: 47}
Testing {8: 31, 16: 34, 0: 34}
#################################
Linear R2 Score Best Feature ['univariate, std(S)', 'univariate, V_at_max(S)', 'univariate, min(dS/dV)', 'vcenter', 'univariate, area(dS/dV)', 'univariate, max(dS/dV)', 'peak curvature']
Linear Percent Error Best Feature ['univariate, std(S)', 'vcenter', 'univariate, area(dS/dV)', 'peak curvature', 'univariate, area(S)', 'univariate, max(dS/dV) - min(dS/dV)', 'univariate, V_at_max(S)']
****************************************************
KNN R2 Score Best Feature ['univariate, max(dS/dV) - min(dS/dV)', 'univariate, V_at_max(S)']
KNN Percent Error Best Feature ['univariate, std(S)', 'univariate, max(S)', 'univariate, min(dS/dV)']
****************************************************
RF R2 Score Best Feature ['univariate, area(dS/dV)', 'univariate, min(dS/dV)', 'univariate, V_at_min(dS/dV)']
RF Percent Error Best Feature ['univariate, max(dS/dV) - min(dS/dV)', 'univariate, V_at_min(dS/dV)']
****************************************************
GP R2 Score Best Feature ['univariate, std(S)', 'univariate, V_at_max(dS/dV)', 'univariate, V_at_max(S)']
GP Percent Error Best Feature ['univariate, std(S)', 'univariate, V_at_max(dS/dV)', 'univariate, V_at_max(S)']
****************************************************
########Paired Permutation Test##############
                     Model Comparison  Observed Diff  Diff mean  Diff std  p value
0  Linear--------- univariate, std(S)      19.010316   3.612422  2.717097   0.0000
1                 Linear--------- KNN      14.965660   3.621785  2.659492   0.0003
2                  Linear--------- RF      13.785516   3.562176  2.664851   0.0013
3                  Linear--------- GP      10.143973   3.289923  2.442589   0.0115
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
