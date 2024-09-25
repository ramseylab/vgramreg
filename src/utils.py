import os
import sys
sys.path.insert(0, '../')

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from pycombat import Combat

import matplotlib.pyplot as plt
import seaborn as sns

from typing import List

from src.load_dataset import select_normalizer

def verify_batch_label_dist(y):
    new_df          = pd.DataFrame(y, columns=['y'])
    new_df['batch'] = new_df['y'].apply(lambda x: x.split('_')[0])
    new_df['label'] = new_df['y'].apply(lambda x: x.split('_')[1])
    
    df = new_df.groupby (['batch', 'label']).count()
    return df

def find_adj_score(N: int, P: int, R_2: float) -> float:
    return (1 - (1 - R_2)*(N - 1)/(N - P - 1))

def per_error(y_test:pd.Series, y_pred:np.array, y_LOD:float)->float:
   
    mask           = (y_test != 0)    # Non Zero Concentration
    zero_mask      = ~(mask)          # Zero Concentration

    y_pred         = np.maximum(y_pred, 0.0)

    # Only for non zero concentration
    non_zero_per_error = np.abs(y_test[mask] - y_pred[mask])/(0.5*(y_test[mask] + y_pred[mask]))
   
    # zero concentration
    zero_per_error     = np.abs(y_test[zero_mask] - y_pred[zero_mask]) / y_LOD

    assert not(np.isnan(zero_per_error).any())
    assert not(np.isnan(non_zero_per_error).any())

    per_error         = np.concatenate((non_zero_per_error, zero_per_error))
    per_error         = np.mean(per_error) * 100

    return per_error

def calculate_y_LOD(X_train, y_train):
    model_yLOD = LinearRegression()
    model_yLOD.fit(X_train[['univariate, std(S)']], y_train)          # Selecting standard deviation of sample as a feature
    
    S  = model_yLOD.coef_[0]                                                    # Slope of fitting line y=Sx + c 
    SD = X_train['univariate, std(S)'][(y_train==0).to_numpy()].std() # Standard deviation of S blank

    return 2.636369 * S * SD # We got the constant value from the -qt(0.01/2, 83) there number of blanks = 84 and we are using k-1 degree 84 -1 = 83


def tsen_pca_viz(data:List[pd.DataFrame], batch_labels:List[str], labels:List[str], filename=''):
    
    data        = pd.concat(data)
    tsne        = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(data)
    
    # Convert t-SNE result to a DataFrame for easier plotting
    tsne_df          = pd.DataFrame(data=tsne_result, columns=['t-SNE1', 't-SNE2'])
    tsne_df['Batch'] = batch_labels
    tsne_df['labels'] = labels

    os.makedirs('batch_effect', exist_ok=True)
    
    # Plot t-SNE

    fig, axs = plt.subplots(4, 1, figsize=(15, 12))
    
    sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='Batch', style='labels', data=tsne_df, palette='deep', markers=['o', 's', '^'], s=20, ax=axs[0])
    axs[0].set_title('t-SNE: Batch Effect Visualization')
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    

    
    # Perform PCA
    pca        = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    
    # Convert PCA result to a DataFrame for easier plotting
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['Batch']  = batch_labels
    pca_df['labels'] = labels
    
    # Plot PCA
    sns.scatterplot(x='PC1', y='PC2', hue='Batch', style='labels', data=pca_df, palette='deep', markers=['o', 's', '^'], s=20, ax=axs[1])
    axs[1].set_title('PCA: Batch Effect Visualization')
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.subplot(4,1,3)
    sns.kdeplot(data=pca_df, x='PC1', hue='Batch', fill=True, common_norm=False, palette='deep', alpha=0.5)
    # axs[2].set_title('Density Plot')
    axs[2].set_xlabel('PCA Component Value')
    axs[2].set_ylabel('Density')

    plt.subplot(4,1,4)
    sns.kdeplot(data=pca_df, x='PC2', hue='Batch', fill=True, common_norm=False, palette='deep', alpha=0.5)
    # axs[3].set_title('Density Plot')
    axs[3].set_xlabel('PCA Component Value')
    axs[3].set_ylabel('Density')

    if filename!='': plt.savefig(f'batch_effect/{filename}.png', dpi=300)
    else: plt.show()

def calculate_per_diff(model:BaseEstimator, X:pd.DataFrame, y:pd.Series, kf:KFold, y_LOD:float) -> np.ndarray:
    per_diff_all = []
    
    for train_index, test_index in kf.split(X):
        model_ = clone(model)
        
        # Split the data into training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]
    
        model_.fit(X_train, y_train)
        
        mask           = (y_test != 0)    # Non Zero Concentration
        zero_mask      = ~(mask)          # Zero Concentration

        y_pred         = model_.predict(X_test)
        y_pred         = np.maximum(y_pred, 0.0)

        # Only for non zero concentration
        non_zero_per_error = np.abs(y_test[mask] - y_pred[mask])/(0.5*(y_test[mask] + y_pred[mask]))
        
        # zero concentration
        zero_per_error     = np.abs(y_test[zero_mask] - y_pred[zero_mask]) / y_LOD

        assert not(np.isnan(zero_per_error).any())
        assert not(np.isnan(non_zero_per_error).any())

        per_error         = np.concatenate((non_zero_per_error, zero_per_error))
        per_error         = np.mean(per_error) * 100

        assert not(np.isnan(per_error)) # To check if any output is invalid or nan
        per_diff_all.append(per_error)

    
    return np.array(per_diff_all).mean()
    

def calculate_r2_score(model:BaseEstimator, X:pd.DataFrame, y:pd.Series, kf:KFold) -> np.ndarray:
    y_pred_all, y_test_all = [], []

    for train_index, test_index in kf.split(X):
        model_ = clone(model)
        
        # Split the data into training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]
    
        model_.fit(X_train, y_train)
        
        y_pred         = model_.predict(X_test)
        y_pred         = np.maximum(y_pred, 0.0)

        y_pred_all += y_pred.tolist()
        y_test_all += y_test.tolist()

    score         = r2_score(y_test_all, y_pred_all)
    adj_score     = find_adj_score(len(y_pred_all), X_train.shape[1], score) # N, P, R2 score

    return np.array(score), np.array(adj_score)
    
def combine_all_batches(data, dataset_name):
    features       = pd.concat(data)
    batch_labels   = np.repeat(dataset_name, repeats=[len(i) for i in data])

    return features, batch_labels

def split_batches_back(data, combat_output):
    ind_concat    = []
    temp_ind      = 0
    for i in range(len(data)):
        ind_concat.append((temp_ind, temp_ind +  data[i].shape[0]))
        temp_ind      = ind_concat[i][-1]

    output = [pd.DataFrame(combat_output[i:j], columns=data[0].columns) for i, j in ind_concat]

    for i, x in enumerate(output):
        assert x.shape == data[i].shape
    return output

def perform_combat_normalization(data:List[pd.DataFrame], dataset_name:List[str]) -> List[pd.DataFrame]:
    
    features, batch_labels = combine_all_batches(data, dataset_name)
    
    combat        = Combat()
    combat.fit(features.values, batch_labels)
    combat_output = combat.transform(features.values, batch_labels)

    output = split_batches_back(data, combat_output)

    return output, batch_labels, combat



def normalizer_inference_dataset(dataset, normalizer_type='mean_std'):
    X = dataset.drop(columns='file').copy()
    columns = X.columns
    y = dataset['file'].apply(lambda x: int(x.split('_')[-2].replace('cbz','')))

    scaler = select_normalizer(normalizer_type)
    X      = scaler.fit_transform(X)
    X      = pd.DataFrame(X, columns=columns)

    X.rename(columns={"PH": 'univariate, max(S)', 'signal_std':'univariate, std(S)', 'signal_mean':'univariate, mean(S)', 'peak area':'univariate, area(S)', \
                        'dS_dV_area':'univariate, area(dS/dV)', 'dS_dV_max_peak':'univariate, max(dS/dV)', 'dS_dV_min_peak':'univariate, min(dS/dV)',\
                    'dS_dV_peak_diff':'univariate, max(dS/dV) - min(dS/dV)', \
                    'peak V':'univariate, V_max(S)', 'dS_dV_max_V':'univariate, V_max(dS/dV)', 'dS_dV_min_V':'univariate, V_min(dS/dV)',\
        }, inplace = True)
    
    return X, y

