import pandas as pd
import numpy as np
import seaborn as sns
import os

import matplotlib.pyplot as plt

from typing import Tuple

# Find outliers 
def find_outliers_remove(feature_select_outlier:str,
                  df_data: pd.DataFrame,
                  save_path='') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ This functions finds outliers in the input dataset 
        with respect to each labels and remove the outliers from the dataset
    """
    labels   = sorted(df_data['label'].unique())
    outliers = {i:{'above':[], 'below':[]} for i in labels}

    if save_path != '' :
        plt.figure(figsize=(7,4))
        sns.boxplot(df_data, x='label', y=feature_select_outlier)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    for i in labels:
       
        Q1  = np.percentile(df_data[df_data['label']==i][feature_select_outlier], q=25)
        Q3  = np.percentile(df_data[df_data['label']==i][feature_select_outlier], q=75)
        
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
    
        above_upper_bound = df_data[[feature_select_outlier, 'file']][(df_data[feature_select_outlier]>upper_bound) & (df_data['label']==i)]
        above_lower_bound = df_data[[feature_select_outlier, 'file']][(df_data[feature_select_outlier]<lower_bound) & (df_data['label']==i)]

        df_data = df_data[((df_data[feature_select_outlier]<=upper_bound)&\
                          (df_data[feature_select_outlier]>=lower_bound)&\
                          (df_data['label']==i)) | df_data['label']!=i]
        
        outliers[i]['above'] += above_upper_bound['file'].apply(lambda x: x.split('/')[-1]).to_list()
        outliers[i]['below'] += above_lower_bound['file'].apply(lambda x: x.split('/')[-1]).to_list()

    

    return df_data, outliers