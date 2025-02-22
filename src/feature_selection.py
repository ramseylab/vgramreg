
import pandas as pd
import numpy as np
import pickle

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, GridSearchCV

from typing import Tuple

from src.load_models import select_model
from src.utils import find_adj_score, calculate_y_LOD, calculate_r2_score, calculate_per_diff, calculate_combined_r2_and_per_diff, calculate_per_diff_KFold, calculate_r2_score_KFold, calculate_combined_r2_and_per_diff_KFold
from src.config import PARAMS_GRID

class ModelSelection():
    def __init__(self, model_name:str, X_train:pd.DataFrame, y_train:pd.Series):
        self.X_train, self.y_train = X_train, y_train
        self.model = select_model(model_name)
        self.model_name = model_name
        self.all_feature_scores = []
    
        self.y_LOD = calculate_y_LOD(self.X_train, self.y_train) 

    def save(self, path:str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.model, path) 

    def find_score(self, 
                   y_true:np.array, 
                   y_pred:np.array, 
                   y_LOD:np.float64,
                   metric:str, 
                   use_adjusted_r2:bool,
                   feature_len:int):
        
        if metric == 'r2':
            score = calculate_r2_score(y_true, y_pred)
            if use_adjusted_r2:
                score = find_adj_score(len(y_true), feature_len, score)
            return score
        
        elif metric == 'per_diff': return calculate_per_diff(y_true, y_pred, y_LOD)
        else: return calculate_combined_r2_and_per_diff(y_true, y_pred, y_LOD)

    def find_score_KFold(self, 
                   kf:KFold, 
                   features:list, 
                   metric:str,
                   use_adjusted_r2:bool) -> np.ndarray:
        
        if 'r2'in metric:
            return calculate_r2_score_KFold(self.model, self.X_train[features], self.y_train, kf, use_adjusted_r2=use_adjusted_r2)

        elif metric=='per_diff':  
            return calculate_per_diff_KFold(self.model, self.X_train[features], self.y_train, self.y_LOD, kf)

        else:
            return calculate_combined_r2_and_per_diff_KFold(self.model, self.X_train[features], self.y_train, kf, self.y_LOD, use_adjusted_r2=use_adjusted_r2)

    
    def find_per_diff(self, kf:KFold, features:list) -> np.ndarray:
        return np.array(calculate_per_diff_KFold(self.model, self.X_train[features], self.y_train, self.y_LOD, kf))
    
    def fit(self, features:list) -> None:
        self.model.fit(self.X_train[features], self.y_train)

    def list_all_parameters(self, params_dict:dict) -> pd.DataFrame:
        all_keys     = list(params_dict.keys())
        all_params   = pd.DataFrame(params_dict[all_keys[0]], columns=[all_keys[0]])

        for key in all_keys[1:]:
            all_params = all_params.merge(pd.DataFrame(params_dict[key], columns=[key]), how='cross')

        return all_params
    
    def calculate_score_hyperparameter(self, 
                                       params:dict, 
                                       metric:str, 
                                       features:list, 
                                       kf:KFold, 
                                       use_adjusted_r2:bool) -> np.float64:
        
        # Assign new parameters to the base model
        params_dict = params.to_dict()
        self.model.set_params(**params_dict)

        # Calcualte metric
        if 'r2'in metric:
            return calculate_r2_score_KFold(self.model, self.X_train[features], self.y_train, kf, use_adjusted_r2=use_adjusted_r2)

        elif metric=='per_diff':  
            return calculate_per_diff_KFold(self.model, self.X_train[features], self.y_train, self.y_LOD, kf)

        else:
            return calculate_combined_r2_and_per_diff_KFold(self.model, self.X_train[features], self.y_train, kf, self.y_LOD, use_adjusted_r2=use_adjusted_r2)


        

    def find_best_score_hyperparameter_KFold(self, 
                                  kf:KFold, 
                                  testing_feature:list, 
                                  metric:str, 
                                  use_adjusted_r2:bool) -> Tuple:
        
        all_params_df = self.list_all_parameters(PARAMS_GRID[self.model_name])
        all_params_df['score'] = all_params_df.apply(self.calculate_score_hyperparameter, axis=1, args=(metric, testing_feature,kf,use_adjusted_r2))

        if metric=='per_diff': best_features = all_params_df.loc[all_params_df['score'].idxmin()]
        else: best_features = all_params_df.loc[all_params_df['score'].idxmax()]
        best_score    = best_features['score']
        return best_features.drop('score').to_dict(), best_score


    def find_best_features(self, 
                           kf:KFold, 
                           metric='r2', 
                           use_adjusted_r2=False, 
                           tune_hyperparameter=True) -> list:
        # model = clone(self.model)

        all_features            = self.X_train.columns.values
        self.selected_features  = []
        self.all_feature_scores = []
        self.all_best_params    = []

        self.best_score        = 100.0 if metric=='per_diff' else 0
        flag                   = False
         
        while len(self.selected_features) != len(all_features):
            one_line_score       = []
            one_line_features    = []
            one_line_best_params = []
            
            for feature in all_features:
                
                if feature not in self.selected_features:
                    testing_feature = self.selected_features + [feature]

                    if tune_hyperparameter:
                        best_param, score = self.find_best_score_hyperparameter_KFold(kf, testing_feature, metric, use_adjusted_r2)
                        one_line_best_params.append(best_param)

                    else:
                        score = self.find_score_KFold(kf, testing_feature, metric, use_adjusted_r2)
        
                    one_line_score.append(score)
                    one_line_features.append(feature)
           
            # one_line_score = np.array(one_line_score) if r2_score else one_line_score
            
            if metric=='per_diff':
                best_socre_ind, one_line_best_score = np.argmin(one_line_score), np.min(one_line_score)

            else:
                best_socre_ind, one_line_best_score = np.argmax(one_line_score), np.max(one_line_score)
                

            sel_one_line_feature    = one_line_features[best_socre_ind] 

            if tune_hyperparameter: sel_one_best_params     = one_line_best_params[best_socre_ind]

            temp = {}
            for key, score in zip(one_line_features, one_line_score):
                key = self.selected_features + [key]
                temp[str(key)] = score
                
            if metric=='per_diff':

                if one_line_best_score <= self.best_score:
                    self.best_score = one_line_best_score
                    self.selected_features.append(sel_one_line_feature)
                    if tune_hyperparameter: self.all_best_params.append(sel_one_best_params)
                    self.all_feature_scores.append(temp)

                    flag = False

                else: flag = True
 
            else:
                if one_line_best_score >= self.best_score:
                    self.best_score = one_line_best_score
                    self.selected_features.append(sel_one_line_feature)
                    if tune_hyperparameter: self.all_best_params.append(sel_one_best_params)
                    self.all_feature_scores.append(temp)
                    flag = False

                else: flag = True
                
            if flag: break

        return self.all_feature_scores, self.all_best_params
    
    
    def find_testing_score(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[list, list]:

        # Fit the model with selected features
        self.model.fit(self.X_train[self.selected_features], self.y_train)

        # Return both training and testing r2 score
        return self.model.score(self.X_train[self.selected_features], self.y_train), \
               self.model.score(X_test[self.selected_features], y_test)