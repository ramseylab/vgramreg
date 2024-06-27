import numpy  as np
import pandas as pd

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

from ML_testing import select_model
from graph_visualization import visualize_highest_score_feature_selection, feature_selection_tabularize, visualization_testing_dataset
from config import *

from sklearn.metrics import r2_score

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

    print("######Data Distribution:#########")
    print("Training", find_concentration_distribution(y_train))
    print("Testing",  find_concentration_distribution(y_test))
    print("#################################")
    return X_train, X_test, y_train, y_test

def select_features(X_train, y_train, model_names):
    #model_names    = ['Linear', 'KNN', 'RF', 'GP']

    feature_selection_r2score     = {}
    feature_selection_per_diff    = {}

    kf        = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name in model_names:
        dataset_model_name = model_name
       
        model = ModelSelection(model_name, X_train, y_train)

        feature_selection_r2score[dataset_model_name]  = model.find_best_features(kf, r2_score=True)
        models_features_r2[model_name] = model.selected_features

        feature_selection_per_diff[dataset_model_name] = model.find_best_features(kf, r2_score=False)
        models_features_per[model_name] = model.selected_features

        print(f"{model_name} R2 Score Best Feature",      models_features_r2[model_name])
        print(f"{model_name} Percent Error Best Feature", models_features_per[model_name])

        print("****************************************************")

    return feature_selection_r2score, feature_selection_per_diff

def per_error(y_test, y_pred, y_LOD):
    mask           = (y_test != 0)    # Non Zero Concentration
    zero_mask      = ~(mask)          # Zero Concentration

    y_pred         = np.maximum(y_pred, 0.0)
    # y_LOD          = 1.6193237802284837

    # Only for non zero concentration
    non_zero_per_error = np.abs(y_test[mask] - y_pred[mask])/(0.5*(y_test[mask] + y_pred[mask]))
   
    # zero concentration
    zero_per_error     = np.abs(y_test[zero_mask] - y_pred[zero_mask]) / y_LOD

    assert not(np.isnan(zero_per_error).any())
    assert not(np.isnan(non_zero_per_error).any())

    per_error         = np.concatenate((non_zero_per_error, zero_per_error))
    per_error         = np.mean(per_error) * 100

    return per_error


def find_performance_metric(model_names, r2_top):
    r2_scores  = {'Models':[], 'Scores':[]}
    per_errors = {'Models':[], 'Scores':[]}

    model_names = model_names if not(only_one_multivariate) else r2_top['Models'].values.tolist()

    # model_names = list(set(model_names))

    for model_name in tqdm(model_names):
        # print(model_name)
        model_name = 'Linear' if ((model_name == 'multivariate')) else model_name

        model_r2   = ModelSelection(model_name, X_train, y_train)
        model_per  = ModelSelection(model_name, X_train, y_train)
        
        model_r2.fit(models_features_r2[model_name])
        model_per.fit(models_features_per[model_name])

        y_pred_r2  = model_r2.model.predict(X_test[models_features_r2[model_name]])
        y_pred_per = model_per.model.predict(X_test[models_features_per[model_name]])

        model_per_error      = per_error(y_test, y_pred_per, model_per.y_LOD)
        model_r2_scores      = r2_score(y_test, y_pred_r2)

        # name = name_conversion[model_name]
        # if not (only_one_multivariate) and (model_name == 'Linear'): name = 'Linear'
        model_name = 'multivariate' if ((model_name == 'Linear') and only_one_multivariate) else model_name

        r2_scores['Models'].append(model_name) 
        per_errors['Models'].append(model_name)
        
        r2_scores['Scores'].append(model_r2_scores)
        per_errors['Scores'].append(model_per_error)

    return r2_scores, per_errors


if __name__ == '__main__':

    # Lists of dataset paths
    datasets = ['/Users/sangam/Desktop/Epilepsey/Code/Signal_Analysis/Dataset/2024_02_19_ML1/signal_log_NOrecenter_0.006_0_1.04_0.15_0.17extra_features.xlsx',
           '/Users/sangam/Desktop/Epilepsey/Code/Signal_Analysis/Dataset/2024_02_22_ML2/signal_log_NOrecenter_0.006_0_1.04_0.15_0.17extra_features.xlsx']

    # Names for each features
    model_names           = ['Linear', 'KNN', 'RF', 'GP']

    # Configuration for Output Visualization
    legends               = True
    only_one_multivariate = True #if ('KNN' in model_names) or ('RF' in model_names) or ('GP' in model_names) else True

    # path to save outputs
    output_path_name         = 'Outputs'
    output_path_feature_list = 'Outputs/feature_selection_list'

    os.makedirs(output_path_feature_list, exist_ok=True)

    X_train, X_test, y_train, y_test = load_dataset(datasets)
    feature_selection_r2score, feature_selection_per_diff = select_features(X_train, y_train, model_names)

    # print(X_train.columns, X_test.columns)
   
    # Generate Excel Table showing performance metrics for each step of feature selection staring from univariate model
    for model in model_names:
        df = feature_selection_tabularize(feature_selection_r2score[model])
        df.to_excel(f'Outputs/feature_selection_list/feature_selection_r2score_{model}.xlsx', index=False)

        df = feature_selection_tabularize(feature_selection_per_diff[model])
        df.to_excel(f'Outputs/feature_selection_list/feature_selection_per_error_{model}.xlsx', index=False)

    for only_one_multivariate in [True, False]:
        comparision_model = 'uni_multivariate' if only_one_multivariate else 'linear_nonlinear'
        r2_top = visualize_highest_score_feature_selection(feature_selection_r2score, f"{output_path_name}/{comparision_model}_5_fold_r2score.png",    model_name_conversion, only_one_multivariate=only_one_multivariate, legends=False)
        visualize_highest_score_feature_selection(feature_selection_per_diff, f"{output_path_name}/{comparision_model}_5_fold_per_error.png", model_name_conversion, r2_score=False, only_one_multivariate=only_one_multivariate, legends=True)

        # print(r2_top['Models'].values.tolist())
        test_r2_scores, test_per_errors = find_performance_metric(model_names, r2_top)

        # print(test_r2_scores, test_per_errors)
        visualization_testing_dataset(test_r2_scores,  f'{output_path_name}/{comparision_model}_testing_r2_score.png',  r2_score=True,  only_one_multivariate=only_one_multivariate, legends=False)
        visualization_testing_dataset(test_per_errors, f'{output_path_name}/{comparision_model}_testing_per_error.png', r2_score=False, only_one_multivariate=only_one_multivariate, legends=True)