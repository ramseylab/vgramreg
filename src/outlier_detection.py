import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from typing import Tuple

# Find outliers 
def find_outliers_remove(feature_select_outlier:str,
                  df_data: pd.DataFrame,
                  save_path='') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """This functions finds outliers in the input dataset with respect to each labels and remove the outliers from the dataset
    """
    labels = sorted(df_data['y'].unique())

    for i in labels:
        Q1  = np.percentile(df_data[df_data['y']==i][feature_select_outlier], q=25)
        Q3  = np.percentile(df_data[df_data['y']==i][feature_select_outlier], q=75)
        
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
    
        above_upper_bound = df_data[[feature_select_outlier, 'filename']][(df_data[feature_select_outlier]>upper_bound) & (df_data['y']==i)]
        above_lower_bound = df_data[[feature_select_outlier, 'filename']][(df_data[feature_select_outlier]<lower_bound) & (df_data['y']==i)]

        df_data = df_data[(df_data[feature_select_outlier]<=upper_bound)&\
                          (df_data[feature_select_outlier]>=lower_bound)&\
                          (df_data['y']==i)]

    if save_path !='' :
        plt.figure(figsize=(7,4))
        sns.boxplot(df_data, x='y', y=feature_select_outlier)
        plt.savefig(save_path, dpi=300)

    return df_data, above_lower_bound, above_upper_bound