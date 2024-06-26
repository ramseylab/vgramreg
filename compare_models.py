import numpy  as np
import pandas as pd
from ML_testing import select_model

from sklearn.base import clone
from sklearn.model_selection import KFold

from feature_selection import ModelSelection, visualize_feature_selection, visualize_testing_model, visualize_highest_score_feature_selection
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
import random
import os

from sklearn.preprocessing import StandardScaler
import matplotlib.lines as mlines

# Ignore all warnings
warnings.filterwarnings('ignore')

def find_concentration_distribution(y):
    all_labels = y.tolist()
    unique_    = set(all_labels)
    count = {}
    for i in unique_: count[i] = all_labels.count(i)
    
    return count

def load_dataset(datasets):
    
    df1 = pd.read_excel(datasets[0])
    df2 = pd.read_excel(datasets[1])

    df  = pd.concat([df1, df2])
    df   
    X   = df[["peak area", "peak curvature", "peak V", "vcenter", "PH", "signal_mean", "signal_std", \
                                "dS_dV_max_peak", "dS_dV_min_peak", "dS_dV_peak_diff", "dS_dV_max_V", "dS_dV_min_V", "dS_dV_area"]]
    y   = df['file'].apply(lambda x: int(x.split('_')[-2].replace('cbz','')))

    X.rename(columns={"PH": 'univariate, max(S)', 'signal_std':'univariate, std(S)', 'signal_mean':'univariate, mean(S)', 'peak area':'univariate, area(S)', \
                        'dS_dV_area':'univariate, area(dS/dV)', 'dS_dV_max_peak':'univariate, max(dS/dV)', 'dS_dV_min_peak':'univariate, min(dS/dV)',\
                    'dS_dV_peak_diff':'univariate, max(dS/dV) - min(dS/dV)', \
                    'peak V':'univariate, V_at_max(S)', 'dS_dV_max_V':'univariate, V_at_max(dS/dV)', 'dS_dV_min_V':'univariate, V_at_min(dS/dV)',\
        }, inplace = True)

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler to your data
    scaler.fit(X)

    # Transform the data
    X_normalized = scaler.transform(X)
    
    # Transform the data
    X_normalized = scaler.transform(X)

    # Convert the numpy array back to a DataFrame
    X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

    # Split the total dataset into training (70%) and testing (30%) dataset
    X_train, X_test, y_train, y_test  = train_test_split(X_normalized, y, test_size=0.4, shuffle=True, random_state=20, stratify=y)

    print("Data Distribution:")
    print("Training", find_concentration_distribution(y_train))
    print("Testing",  find_concentration_distribution(y_test))
    return X_train, X_test, y_train, y_test

def select_features(X_train, y_train, model_names):
    #model_names    = ['Linear', 'KNN', 'RF', 'GP']

    feature_selection_r2score     = {}
    feature_selection_per_diff    = {}

    kf        = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name in tqdm(model_names):
        dataset_model_name = model_name
       
        model = ModelSelection(model_name, X_train, y_train)

        feature_selection_r2score[dataset_model_name]  = model.find_best_features(kf, r2_score=True)
        feature_selection_per_diff[dataset_model_name] = model.find_best_features(kf, r2_score=False)
        
        print(dataset_model_name)
        print(f"{model_name} R2 Score Best Feature",    model.selected_features)
        print(f"{model_name} Percent Error Best Feature", model.selected_features)


if __name__ == '__main__':
    datasets = ['/Users/sangam/Desktop/Epilepsey/Code/Signal_Analysis/Dataset/2024_02_19_ML1/signal_log_NOrecenter_0.006_0_1.04_0.15_0.17extra_features.xlsx',
           '/Users/sangam/Desktop/Epilepsey/Code/Signal_Analysis/Dataset/2024_02_22_ML2/signal_log_NOrecenter_0.006_0_1.04_0.15_0.17extra_features.xlsx']

    name_conversion = {'Linear':'multivariate', 'std':'univariate, std(S)', 'mean':'univariate, mean(S)', 'peak area':'univariate, area(S)', \
                        'dS_dV_area':'univariate, area(dS/dV)', 'dS_dV_max_peak':'univariate, max(dS/dV)', 'dS_dV_min_peak':'univariate, min(dS/dV)',\
                    'peak height':'univariate, max(S)', 'dS_dV_peak_diff':'univariate, max(dS/dV) - min(dS/dV)', \
                    'peak V':'univariate, V_at_max(S)', 'dS_dV_max_V':'univariate, V_at_max(dS/dV)', 'dS_dV_min_V':'univariate, V_at_min(dS/dV)',\
                    'vcenter':'vcenter', 'peak curvature': 'peak curvature'}
    
    reverse_name_conversion = {}

    for i in name_conversion:
        reverse_name_conversion[name_conversion[i]] = i
        
    model_name_conversion = {'Linear':'Linear', 'KNN':'KNN', 'RF':'Random Forest', 'GP':'Gaussian Process'}

    model_names           = ['Linear', 'KNN', 'RF', 'GP']

    X_train, X_test, y_train, y_test = load_dataset(datasets)
    select_features(X_train, y_train, model_names)


 