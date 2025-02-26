import pandas as pd
import numpy as np

from typing import Tuple
from sklearn.metrics import r2_score

from src.config import paired_test
from src.utils import per_error
from src.feature_selection import ModelSelection

def pair_permutation_test(model1_pred: np.array, 
                          model2_pred: np.array, 
                          ground_truth:np.array, 
                          y_LOD: float) -> Tuple[float, float, float, float, float]:
    """
        Calculates Statistical Significance level p-value for the given pair
    """

    # Calculate percent error of the models
    model1_scores      = per_error(ground_truth, model1_pred, y_LOD)
    model2_scores      = per_error(ground_truth, model2_pred, y_LOD)

    r2_model1_score, r2_model2_score    = r2_score(ground_truth, model1_pred), r2_score(ground_truth, model2_pred)
    observed_r2_score  = np.abs(r2_model1_score - r2_model2_score)
    
    # Calcualte Observed Difference
    per_diff           = model1_scores - model2_scores
    observed_statistic = np.abs(per_diff)
    
    # Number of permutations
    n_permutations = 10000
    permutation_statistics = np.zeros(n_permutations)
    
    # Permutation process
    for i in range(n_permutations):
        model1_pred_temp = model1_pred.copy()
        model2_pred_temp = model2_pred.copy()

        # Shuffle the prediction values
        random_indexs                    = np.random.random(size=model1_pred.shape[0]) < 0.5
        model1_pred_temp[random_indexs]  = model2_pred[random_indexs]
        model2_pred_temp[random_indexs]  = model1_pred[random_indexs]

        permutation_statistics[i]        = np.abs(per_error(ground_truth, model1_pred_temp, y_LOD) - per_error(ground_truth, model2_pred_temp, y_LOD))

    # Calculate p-value
    p_value = np.mean(permutation_statistics >= observed_statistic)

    return observed_r2_score, per_diff, p_value, permutation_statistics.mean(), permutation_statistics.std()


def find_paired_permutation_test(dataset:Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
                                models_features_per: dict) -> pd.DataFrame:
    """
        Calculates Statistical Significance level p-value for all the given pairs
    """

    df      = pd.DataFrame(columns=['Model Comparison', 'Observed Diff', 'Diff mean', 'Diff std', 'p value'])
    (X_train, X_test, y_train, y_test) = dataset

    for model1_name, model2_name in paired_test: 
        # Load Models
        model1  = ModelSelection(model1_name, X_train, y_train)
        model2  = ModelSelection(model2_name, X_train, y_train)

        # Fit model with the best features
        model1.fit(models_features_per[model1_name])
        model2.fit(models_features_per[model2_name])

        # Predict concentration on testing dataset
        model1_pred = model1.model.predict(X_test[models_features_per[model1_name]])
        model2_pred = model2.model.predict(X_test[models_features_per[model2_name]])

        # Peform permutation test
        _, observed_diff, p_value, diff_mean, diff_std = pair_permutation_test(model1_pred, model2_pred, y_test, model1.y_LOD)

        temp = pd.DataFrame.from_dict({'Model Comparison':[f"{model1_name}--------- {model2_name}"], \
                                    'Observed Diff':[observed_diff], \
                                    'Diff mean':diff_mean, 'Diff std':diff_std,'p value':[p_value]})
        
        df   = pd.concat([df, temp], ignore_index=True)

    return df