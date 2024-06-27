import os
import math
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.ML_testing import select_model

class ModelSelection():
    def __init__(self, model_name, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train
        self.model = select_model(model_name)
        self.all_feature_scores = []

        model_yLOD = LinearRegression()
        model_yLOD.fit(self.X_train[['univariate, std(S)']], self.y_train) # Selecting standard deviation of sample as a feature
      
        S  = model_yLOD.coef_[0]                          # Slope
        SD = self.X_train['univariate, std(S)'][(self.y_train==0).to_numpy()].std() # Standard deviation of S blank
    
        self.y_LOD = 2.636369 * S * SD # We got the constant value from the -qt(0.01/2, 83) there number of blanks = 84 and we are using k-1 degree 84 -1 = 83
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, path) 

    def find_score(self, kf, features):
        return np.array(self.calculate_r2_score(self.model, self.X_train[features], self.y_train, kf))
    
    def find_per_diff(self, kf, features):
        return np.array(self.calculate_per_diff(self.model, self.X_train[features], self.y_train, kf))
    
    def calculate_per_diff(self, model, X, y, kf):
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

    def calculate_r2_score(self, model, X, y, kf):
        scores = [] #cross_val_score(model, X, y, cv=kf)

        for train_index, test_index in kf.split(X):
            model_ = clone(model)
            
            # Split the data into training and testing sets
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]
        
            model_.fit(X_train, y_train)
            
            y_pred         = model_.predict(X_test)
            y_pred         = np.maximum(y_pred, 0.0)

            score       = r2_score(y_test, y_pred)

            scores.append(score)

        return np.array(scores) 
    
    def fit(self, features):
        self.model.fit(self.X_train[features], self.y_train)

    def find_best_features(self, kf, r2_score):
        model = clone(self.model)
        all_features      = self.X_train.columns.values
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
                

            # print("Best Score", sel_one_line_feature, one_line_best_score)

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
    
    def find_testing_score(self, X_test, y_test):
        # Fit the model with selected features
        self.model.fit(self.X_train[self.selected_features], self.y_train)

        # Return both training and testing r2 score
        return self.model.score(self.X_train[self.selected_features], self.y_train), \
               self.model.score(X_test[self.selected_features], y_test)

def visualize_testing_model(training_testing_scores, output_feature_selection):
    #output_feature_selection = 'Feature_Selection/ML1_ML2'
    os.makedirs(os.path.dirname(output_feature_selection), exist_ok=True)

    plt.figure(figsize=(15, 30))
    new_dict = {}
    new_dict['Models']      =  list(training_testing_scores.keys())

    if len(list(training_testing_scores.values())[0]) == 2:
        new_dict['Train_score'] = np.array(list(training_testing_scores.values()))[:, 0]
        new_dict['Test_score']  = np.array(list(training_testing_scores.values()))[:, 1]

        df = pd.DataFrame.from_dict(new_dict)

        ax = df.plot(x='Models', y=['Train_score', 'Test_score'], kind='bar', legend=False)
        plt.legend(loc='lower right')
        for p in ax.patches:
            ax.annotate(str(round(p.get_height(), 4)), (p.get_x() + p.get_width() / 2., 0.5 * p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points',  rotation='vertical')


    else:
        new_dict['Test_score']  = np.array(list(training_testing_scores.values()))[:,0]
       
        df = pd.DataFrame.from_dict(new_dict)

        ax = df.plot(x='Models', y=['Test_score'], kind='bar', legend=False)
        plt.legend(loc='lower right')
        for p in ax.patches:
            ax.annotate(str(round(p.get_height(), 4)), (p.get_x() + p.get_width() / 2., 0.5 * p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points',  rotation='vertical')

    #plt.tight_layout()
    plt.savefig(f'{output_feature_selection}', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    
def visualize_feature_selection(all_dataset_feature_score, data_path, r2_score=True):
    
    os.makedirs(data_path, exist_ok=True)
    plt.figure(figsize=(35, 20))
    for dataset_name in all_dataset_feature_score:

        num_rows = math.ceil(len(all_dataset_feature_score[dataset_name])/2)
        for ind, all_feature in enumerate(all_dataset_feature_score[dataset_name]):
            feature_   = list(all_feature.keys())
            scores     = list(all_feature.values())
            
            if r2_score:
                colors = ['skyblue' if i != np.argmax(scores) else 'orange' for i in range(len(scores))]

            else:
                colors = ['skyblue' if i != np.argmin(scores) else 'orange' for i in range(len(scores))]

            if ind < 2:
                plt.subplot(1, 2, (ind+1))
            else:
                plt.subplot(1, 2, (ind%2+1))

            plt.barh(feature_ , scores, color=colors)

            title_name = f"{dataset_name}_number_features_{str(ind+1)}"
            
            # Add value annotations to the bars
            for i, value in enumerate(scores):
                plt.text(value, i, str(round(value, 4)), ha='left', va='center')

            plt.title(title_name)
            plt.tight_layout()

            if (ind%2 == 0):
                plt.savefig(f'{data_path}/{title_name}_{ind}.svg', dpi=300, format='svg', bbox_inches='tight')
                plt.clf()


def visualize_highest_score_feature_selection(all_dataset_feature_score, path_name):

    os.makedirs(os.path.dirname(path_name), exist_ok=True)
    plt.figure(figsize=(35, 15))

    dict_ = {}

    dict_['Models']   = []
    dict_['Scores']   = []
    dict_['features'] = []
    
    for dataset_name in all_dataset_feature_score:
        best_score       = 0
        selected_feature = None
        
        for ind, all_feature in enumerate(all_dataset_feature_score[dataset_name]):
            feature_   = list(all_feature.keys())
            scores     = list(all_feature.values())

            max_score, max_ind  = np.max(scores), np.argmax(scores)

            if max_score > best_score:
                selected_feature = feature_[max_ind]
                best_score = max_score

            if ('ML1_ML2_Linear' == dataset_name) and (ind ==0):
                argsort_ = np.argsort(scores)[-4:][::-1]
                for i in argsort_:
                    dict_['Models'].append(f'{dataset_name}_{eval(feature_[i])[0]}')
                    dict_['Scores'].append(scores[i])
                    dict_['features'].append(feature_[i])
        
        dict_['Models'].append(dataset_name)
        dict_['Scores'].append(best_score)
        dict_['features'].append(selected_feature)

    df = pd.DataFrame.from_dict(dict_)

    
    ax = df[['Models', 'Scores']].plot(x='Models', y=['Scores'], kind='bar', legend=False)
    plt.legend(loc='lower right')
    
    for p in ax.patches:
        ax.annotate(str(round(p.get_height(), 4)), (p.get_x() + p.get_width() / 2., 0.5 * p.get_height()),
            ha='center', va='center', xytext=(0, 5), textcoords='offset points',  rotation='vertical')

    plt.tight_layout()
    plt.savefig(f'{path_name}', dpi=300)
    plt.clf()

    return df