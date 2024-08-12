
import pandas as pd
import numpy as np
import pickle

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

from typing import Tuple

from src.load_models import select_model
from src.utils import find_adj_score, calculate_y_LOD, calculate_r2_score, calculate_per_diff

class ModelSelection():
    def __init__(self, model_name:str, X_train:pd.DataFrame, y_train:pd.Series):
        self.X_train, self.y_train = X_train, y_train
        self.model = select_model(model_name)
        self.all_feature_scores = []
    
        self.y_LOD = calculate_y_LOD(self.X_train, self.y_train) 

    def save(self, path:str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.model, path) 

    def find_score(self, kf:KFold, features:list) -> np.ndarray:
        return np.array(calculate_r2_score(self.model, self.X_train[features], self.y_train, kf))
    
    def find_per_diff(self, kf:KFold, features:list) -> np.ndarray:
        return np.array(calculate_per_diff(self.model, self.X_train[features], self.y_train, kf))
    
    def fit(self, features:list) -> None:
        self.model.fit(self.X_train[features], self.y_train)

    def find_best_features(self, kf:KFold, r2_score:float) -> list:
        model = clone(self.model)

        all_features           = self.X_train.columns.values
        self.selected_features = []
        self.all_feature_scores = []

        best_score        = [0, 0] if r2_score else 100.0
        flag              = False

        while len(self.selected_features) != len(all_features):
            one_line_score    = []
            one_line_features = []
            
            for feature in all_features:
                
                if feature not in self.selected_features:
                    testing_feature = self.selected_features + [feature]
                    
                    if r2_score:
                        score = self.calculate_r2_score(model, self.X_train[testing_feature], self.y_train, kf)
                    else:  
                        score = self.calculate_per_diff(model, self.X_train[testing_feature], self.y_train, kf)
                    
                    one_line_score.append(score)
                    one_line_features.append(feature)
           
            one_line_score = np.array(one_line_score) if r2_score else one_line_score
            
            if r2_score==True:
                best_socre_ind      = np.argmax(one_line_score[:,0])
                one_line_best_score = one_line_score[best_socre_ind]

            else:
                best_socre_ind, one_line_best_score = np.argmin(one_line_score), np.min(one_line_score)

            sel_one_line_feature    = one_line_features[best_socre_ind] 

            temp = {}
            for key, score in zip(one_line_features, one_line_score):
                key = self.selected_features + [key]
                temp[str(key)] = score
                
            if r2_score:
                if one_line_best_score[0] > best_score[0]:
                    best_score = one_line_best_score
                    self.selected_features.append(sel_one_line_feature)
                    self.all_feature_scores.append(temp)
                    flag = False

                else: flag = True
                        
            else:
                if one_line_best_score <= best_score:
                    best_score = one_line_best_score
                    self.selected_features.append(sel_one_line_feature)
                    self.all_feature_scores.append(temp)
                    flag = False

                else: flag = True

            if flag: break
        
        self.best_score = best_score
        return self.all_feature_scores
    
    def find_testing_score(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[list, list]:

        # Fit the model with selected features
        self.model.fit(self.X_train[self.selected_features], self.y_train)

        # Return both training and testing r2 score
        return self.model.score(self.X_train[self.selected_features], self.y_train), \
               self.model.score(X_test[self.selected_features], y_test)