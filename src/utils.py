import os

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

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


def tsen_pca_viz(data, batch_labels, labels, filename=''):
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

    plt.savefig(f'batch_effect/{filename}.png', dpi=300)