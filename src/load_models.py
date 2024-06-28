from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from sklearn import svm
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso

import pandas as pd
import numpy as np

def select_model(model_name, C_value=1):
    if model_name == 'DT':
        return  DecisionTreeRegressor()
    
    elif model_name=='RF':
        return RandomForestRegressor()
    
    elif model_name=='KNN':
        return KNeighborsRegressor()
    
    elif 'SVM' in model_name:
        return svm.SVR(C=C_value, kernel='linear')
    
    elif model_name=='GP':
        kernel = RBF(length_scale=1.0) 
        return GaussianProcessRegressor(kernel=kernel, alpha=1.5)

    elif ('Linear' in model_name) or ('univariate' in model_name) or ('multivariate' in model_name) :
        return LinearRegression()
    
    elif model_name=='Lasso':
        return Lasso(alpha=0.1)

    


if __name__ == "__main__":
    dataset = ['Dataset/2024_02_19_ML1/signal_log_NOrecenter_0.006_0_1.04_0.15_0.17.xlsx',
               'Dataset/2024_02_22_ML2/signal_log_NOrecenter_0.006_0_1.04_0.15_0.17.xlsx',
               'Dataset/2024_02_28_ML3/signal_log_NOrecenter_0.006_0_1.04_0.15_0.17.xlsx',
               'Dataset/2024_03_11_ML4/signal_log_NOrecenter_0.006_0_1.04_0.15_0.17.xlsx']
    
    model_names = ['DT', 'RF', 'KNN', 'GP', 'XGB'] + [f'SVM_{i}' for i in [0.1, 1.0, 5.0, 10.0, ]]
    df_all  = pd.DataFrame(columns = ['ML_Dataset', 'Models', 'Score'])
    
    for data_path in dataset:
        dataset_name = data_path.split('/')[1].split('_')[-1]

        df = pd.read_excel(data_path)

        X  = df[['signal', 'peak V', 'vcenter', 'PH']].to_numpy()
        y  = df['file'].apply(lambda x: int(x.split('_')[-2].replace('cbz',''))).to_numpy()


        # Initialize the cross-validation method (e.g., KFold with 5 folds)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)   

        for model_name in model_names:
            C_value = None

            if 'SVM' in model_name:
                C_value = float(model_name.split('_')[-1])

            model = select_model(model_name, C_value)

            # Perform cross-validation
            scores      = cross_val_score(model, X, y, cv=kf) 
            mean_scores = np.mean(scores)  
            
            row         = {'ML_Dataset':[dataset_name], 'Models':[model_name], 'Score':[mean_scores]}
            temp_df     = pd.DataFrame.from_dict(row)

            df_all          = pd.concat([df_all, temp_df], ignore_index=True)
            print(f"{model_name} "," mean_scores:", mean_scores, scores)
        
        row         = {'ML_Dataset':[''], 'Models':[''], 'Score':['']}
        temp_df     = pd.DataFrame.from_dict(row)
        df_all      = pd.concat([df_all, temp_df], ignore_index=True)
    
    df_all.to_excel('output.xlsx', index=False)