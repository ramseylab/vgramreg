
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

class ModelSelection():
    def __init__(self, model_name:str, X_train:pd.DataFrame, y_train:pd.Series):
        self.X_train, self.y_train = X_train, y_train
        self.model = select_model(model_name)
        self.all_feature_scores = []

        model_yLOD = LinearRegression()
        model_yLOD.fit(self.X_train[['univariate, std(S)']], self.y_train)          # Selecting standard deviation of sample as a feature
      
        S  = model_yLOD.coef_[0]                                                    # Slope of fitting line y=Sx + c 
        SD = self.X_train['univariate, std(S)'][(self.y_train==0).to_numpy()].std() # Standard deviation of S blank
    
        self.y_LOD = 2.636369 * S * SD # We got the constant value from the -qt(0.01/2, 83) there number of blanks = 84 and we are using k-1 degree 84 -1 = 83
    
    def save(self, path:str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.model, path) 

    def find_score(self, kf:KFold, features:list) -> np.ndarray:
        return np.array(self.calculate_r2_score(self.model, self.X_train[features], self.y_train, kf))
    
    def find_per_diff(self, kf:KFold, features:list) -> np.ndarray:
        return np.array(self.calculate_per_diff(self.model, self.X_train[features], self.y_train, kf))
    
    def calculate_per_diff(self, model:BaseEstimator, X:pd.DataFrame, y:pd.Series, kf:KFold) -> np.ndarray:
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
            zero_per_error     = np.abs(y_test[zero_mask] - y_pred[zero_mask]) / self.y_LOD

            assert not(np.isnan(zero_per_error).any())
            assert not(np.isnan(non_zero_per_error).any())

            per_error         = np.concatenate((non_zero_per_error, zero_per_error))
            per_error         = np.mean(per_error) * 100

            assert not(np.isnan(per_error)) # To check if any output is invalid or nan
            per_diff_all.append(per_error)

        
        return np.array(per_diff_all) 

    def calculate_r2_score(self, model:BaseEstimator, X:pd.DataFrame, y:pd.Series, kf:KFold) -> np.ndarray:
        scores   = []
        all_pred, all_gt = [], []
        
        for train_index, test_index in kf.split(X):
            model_ = clone(model)
            
            # Split the data into training and testing sets
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]
        
            model_.fit(X_train, y_train)
            
            y_pred         = model_.predict(X_test)
            y_pred         = np.maximum(y_pred, 0.0)

            all_pred       += y_pred.tolist()
            all_gt         += y_test.tolist()
            # score       = r2_score(y_test, y_pred)

            # scores.append(score)
        score       = r2_score(all_pred, all_gt)

        return np.array(score) 
    
    def fit(self, features:list) -> None:
        self.model.fit(self.X_train[features], self.y_train)

    def find_best_features(self, kf:KFold, r2_score:float) -> list:
        model = clone(self.model)

        all_features           = self.X_train.columns.values
        self.selected_features = []
        self.all_feature_scores = []

        best_score        = 0.0 if r2_score else 100.0
        flag              = False

        while len(self.selected_features) != len(all_features):
            one_line_score    = []
            one_line_features = []
            
            for feature in all_features:
                
                if feature not in self.selected_features:
                    testing_feature = self.selected_features + [feature]
                    
                    if r2_score:
                        score = self.calculate_r2_score(model, self.X_train[testing_feature], self.y_train, kf).mean()
                    else:  
                        score = self.calculate_per_diff(model, self.X_train[testing_feature], self.y_train, kf).mean()
                    
                    one_line_score.append(score)
                    one_line_features.append(feature)
           
            if r2_score==True:
                best_socre_ind, one_line_best_score = np.argmax(one_line_score), np.max(one_line_score)

            else:
                best_socre_ind, one_line_best_score = np.argmin(one_line_score), np.min(one_line_score)

            sel_one_line_feature    = one_line_features[best_socre_ind]

            temp = {}
            for key, score in zip(one_line_features, one_line_score):
                key = self.selected_features + [key]
                temp[str(key)] = score
                

            if r2_score:
                if one_line_best_score > best_score:
                    best_score = one_line_best_score
                    self.selected_features.append(sel_one_line_feature)
                    self.all_feature_scores.append(temp)
                    flag = False

                else:
                    flag = True
                        
            else:
                if one_line_best_score <= best_score:
                    best_score = one_line_best_score
                    self.selected_features.append(sel_one_line_feature)
                    self.all_feature_scores.append(temp)
                    flag = False

                else:
                    flag = True

            if flag:
                    break

        return self.all_feature_scores
    
    def find_testing_score(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[list, list]:

        # Fit the model with selected features
        self.model.fit(self.X_train[self.selected_features], self.y_train)

        # Return both training and testing r2 score
        return self.model.score(self.X_train[self.selected_features], self.y_train), \
               self.model.score(X_test[self.selected_features], y_test)