import pandas as pd
import numpy as np

from error_metrics import per_error
from sklearn.metrics import r2_score
from config import paired_test

from feature_selection import ModelSelection

def pair_permutation_test(model1_pred, model2_pred, ground_truth, y_LOD):

    # Calculate observed test statistic
    model1_scores      = per_error(ground_truth, model1_pred, y_LOD)
    model2_scores      = per_error(ground_truth, model2_pred, y_LOD)

    r2_model1_score, r2_model2_score    = r2_score(ground_truth, model1_pred), r2_score(ground_truth, model2_pred)
    observed_r2_score  = np.abs(r2_model1_score - r2_model2_score)
    
    observed_statistic = np.abs(model1_scores - model2_scores) # Observed Difference
    
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

    return observed_r2_score, observed_statistic, p_value, permutation_statistics.mean(), permutation_statistics.std()

def color_rows(row):
    return ['background-color: red' if row['p value'] > 0.05 else '' for _ in row]

def description_row(row):
    if row['p value'] < 0.05:
        return 'Statistically Significant that the Model1 is better than Model2' if row['Observed Diff']>0 else 'Statistically Significant that the Model2 is better than Model1'

    else:
        return 'Observed difference is not statistically significant'
    


def find_paired_permutation_test(dataset, models_features_per, models_features_r2):

    df      = pd.DataFrame(columns=['Model Comparison', 'Observed Diff', 'Diff mean', 'Diff std', 'p value'])
    (X_train, X_test, y_train, y_test) = dataset
    for model1_name, model2_name in paired_test: 
        
        model1  = ModelSelection(model1_name, X_train, y_train)
        model2  = ModelSelection(model2_name, X_train, y_train)

        # Fit model
        model1.fit(models_features_per[model1_name])
        model2.fit(models_features_per[model2_name])

        model1_pred = model1.model.predict(X_test[models_features_per[model1_name]])
        model2_pred = model2.model.predict(X_test[models_features_per[model2_name]])

        _, observed_diff, p_value, diff_mean, diff_std = pair_permutation_test(model1_pred, model2_pred, y_test, model1.y_LOD)

        temp = pd.DataFrame.from_dict({'Model Comparison':[f"{model1_name}--------- {model2_name}"], \
                                    'Observed Diff':[observed_diff], \
                                    'Diff mean':diff_mean, 'Diff std':diff_std,'p value':[p_value]})
        df   = pd.concat([df, temp], ignore_index=True)

    return df