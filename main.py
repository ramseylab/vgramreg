import warnings
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from typing import Tuple

# Import local files
from src.graph_visualization import visualize_highest_score_feature_selection, feature_selection_tabularize, visualization_testing_dataset
from src.config import *
from src.utils import per_error
from src.permutation_test import find_paired_permutation_test
from src.feature_selection import ModelSelection
from src.load_dataset import load_dataset

# Ignore all warnings
warnings.filterwarnings('ignore')

def select_features(X_train: pd.DataFrame, y_train: pd.DataFrame, model_names: list) -> Tuple[dict, dict]:
    """
        This function selects the best feature combinations for both 
        R2 score and percent error metrics for the given lists of models
    """

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


def find_performance_metric(model_names: list, r2_top:pd.DataFrame) -> Tuple[dict, dict]:
    """
        Calcualte R2 Score and Percent Error on testing dataset
    """

    r2_scores  = {'Models':[], 'Scores':[]}
    per_errors = {'Models':[], 'Scores':[]}

    model_names = model_names if not(only_one_multivariate) else r2_top['Models'].values.tolist()

    for model_name in model_names:
        
        model_name = 'Linear' if ((model_name == 'multivariate')) else model_name

        model_r2   = ModelSelection(model_name, X_train, y_train)
        model_per  = ModelSelection(model_name, X_train, y_train)
        
        model_r2.fit(models_features_r2[model_name])
        model_per.fit(models_features_per[model_name])

        y_pred_r2  = model_r2.model.predict(X_test[models_features_r2[model_name]])
        y_pred_per = model_per.model.predict(X_test[models_features_per[model_name]])

        model_per_error      = per_error(y_test, y_pred_per, model_per.y_LOD)
        model_r2_scores      = r2_score(y_test, y_pred_r2)

        model_name = 'multivariate' if ((model_name == 'Linear') and only_one_multivariate) else model_name

        r2_scores['Models'].append(model_name) 
        per_errors['Models'].append(model_name)
        
        r2_scores['Scores'].append(model_r2_scores)
        per_errors['Scores'].append(model_per_error)

    return r2_scores, per_errors


if __name__ == '__main__':

    # Name of Models to test
    model_names           = ['Linear', 'KNN', 'RF', 'GP']

    # Configuration for Visualization
    legends               = True
    only_one_multivariate = True 

    # path to save outputs
    output_path_name         = 'Outputs'
    output_path_feature_list = 'Outputs/feature_selection_list'

    os.makedirs(output_path_feature_list, exist_ok=True)

    # Load Training Dataset
    X_train, X_test, y_train, y_test = load_dataset()
    
    # Select the best features for each performance mertrics R2 score and Percent Error
    feature_selection_r2score, feature_selection_per_diff = select_features(X_train, y_train, model_names)

    # Perform Paired wise permutation test
    dataset = (X_train, X_test, y_train, y_test)
    permutation_test = find_paired_permutation_test(dataset, models_features_per)
   
    # Generate Excel Table showing performance metrics for each step of feature selection staring from univariate model
    for model in model_names:
        df = feature_selection_tabularize(feature_selection_r2score[model])
        df.to_excel(f'Outputs/feature_selection_list/feature_selection_r2score_{model}.xlsx', index=False)

        df = feature_selection_tabularize(feature_selection_per_diff[model])
        df.to_excel(f'Outputs/feature_selection_list/feature_selection_per_error_{model}.xlsx', index=False)

    # Comparison between the models
    for only_one_multivariate in [True, False]:
        comparision_model = 'uni_multivariate' if only_one_multivariate else 'linear_nonlinear'

        # Plot the R2 score and Percent Error in the Bar chart on 5-fold cross-validation training dataset
        r2_top = visualize_highest_score_feature_selection(feature_selection_r2score, f"{output_path_name}/{comparision_model}_5_fold_r2score.png",    model_name_conversion, only_one_multivariate=only_one_multivariate, legends=False)
        visualize_highest_score_feature_selection(feature_selection_per_diff, f"{output_path_name}/{comparision_model}_5_fold_per_error.png", model_name_conversion, r2_score=False, only_one_multivariate=only_one_multivariate, legends=True)

        # Calculate R2 Score and Percent Error on Testing Dataset
        test_r2_scores, test_per_errors = find_performance_metric(model_names, r2_top)

        # Plot the R2 score and Percent Error in the Bar chart
        visualization_testing_dataset(test_r2_scores,  f'{output_path_name}/{comparision_model}_testing_r2_score.png',  r2_score=True,  only_one_multivariate=only_one_multivariate, legends=False)
        visualization_testing_dataset(test_per_errors, f'{output_path_name}/{comparision_model}_testing_per_error.png', r2_score=False, only_one_multivariate=only_one_multivariate, legends=True)

    print("########Paired Permutation Test##############")
    print(permutation_test)