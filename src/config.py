from sklearn.gaussian_process.kernels import Matern, RBF

DATASET_PATH = '/Users/sangam/Desktop/Epilepsey/Code/vgramreg/dataset/ML4'
OUTPUT_PATH  = 'vgramreg/results/Journal_paper/'

name_conversion = {'Linear':'multivariate', 'std':'univariate, std(S)', 'mean':'univariate, mean(S)', 'peak area':'univariate, area(S)', \
                        'dS_dV_area':'univariate, area(dS/dV)', 'dS_dV_max_peak':'univariate, max(dS/dV)', 'dS_dV_min_peak':'univariate, min(dS/dV)',\
                    'peak height':'univariate, max(S)', 'dS_dV_peak_diff':'univariate, max(dS/dV) - min(dS/dV)', \
                    'peak V':'univariate, V_max(S)', 'dS_dV_max_V':'univariate, V_max(dS/dV)', 'dS_dV_min_V':'univariate, V_min(dS/dV)',\
                    'vcenter':'vcenter', 'peak curvature': 'peak curvature', 'KNN':'KNN', 'RF':'Random Forest', 'GP':'Gaussian Process'}
    
reverse_name_conversion = {}

for i in name_conversion:
    reverse_name_conversion[name_conversion[i]] = i
    
model_name_conversion = {'Linear':'Linear', 'KNN':'KNN', 'RF':'Random Forest', 'GP':'Gaussian Process', 'multivariate':'multivariate', 'SVM':'SVM', 'Lasso':'Lasso', 'Ridge':'Ridge'}

# Set best R2 feature for R2 metric
models_features_r2 = {
                'KNN':    ['min(dS/dV)', 'max(S)'], 
                'Linear': ['min(dS/dV)', 'V_max(S)', 'f2', 'f1', 'V_max(dS/dV)'], 
                'RF':     ['min(dS/dV)', 'area(S)', 'f2', 'V_max(dS/dV)'], 
                'SVM':    ['min(dS/dV)', 'area(S)', 'vcenter', 'V_max(dS/dV)', 'max(dS/dV)'], 
                'GP':     ['min(dS/dV)', 'area(S)', 'V_max(dS/dV)', 'max(dS/dV)']                 
                }

models_features_per = models_features_r2.copy()

# Set best feature for per error metric
models_features_per.update({
                'KNN':    ['min(dS/dV)', 'max(S)'], 
                'Linear': ['min(dS/dV)', 'V_max(S)', 'f2', 'f1', 'V_max(dS/dV)'], 
                'RF':     ['min(dS/dV)', 'area(S)', 'f2', 'V_max(dS/dV)'], 
                'SVM':    ['min(dS/dV)', 'area(S)', 'vcenter', 'V_max(dS/dV)', 'max(dS/dV)'], 
                'GP':     ['min(dS/dV)', 'area(S)', 'V_max(dS/dV)', 'max(dS/dV)']
                })

paired_test = [('KNN', 'Linear'),
               ('KNN', 'SVM'), 
               ('KNN', 'GP'),
               ('RF', 'GP'),
               ('RF', 'Linear'),
               ('RF', 'RF'),
               ('univariate, max(dS/dV)', 'GP'),
               ('Linear', 'KNN'),
               ('Linear', 'RF'), 
               ('Linear', 'GP'),
               ('Linear', 'SVM'),
               ('Linear', 'Ridge'),
               ('Linear', 'Lasso'),
               ('KNN', 'SVM')
              ]

#### Hyperparamters With Norm ############
# Initial params
params = {
    'SVM': {'C': 250, 
            'gamma': 0.1, 
            'kernel': 'rbf'},
    'RF': {'n_estimators': 40, 
           'max_depth': 5, 
           'min_samples_split': 4, 
           'min_samples_leaf': 1, 
           'max_features': 2},
    'KNN': {'n_neighbors': 12, 
            'weights': 'distance', 
            'metric': 'manhattan'},
   'GP': {'kernel': 1**2 * Matern(length_scale=1, nu=1.5), 
          'alpha': 2.5},
    'Linear': {}
}


# Parameter Grids
PARAMS_GRID = {'SVM':{
                    'C': [15.0, 20.0, 100.0, 150.0, 200.0, 250.0, 300.0],
                    'gamma': [0.005, 0.01, 0.03, 0.1, 1.0, 1.5],
                    'kernel': ['rbf', 'sigmoid']},
              
              'RF': {
                    'n_estimators': [20, 40, 60, 80],
                    'max_depth': [2, 4, 5, 10],
                    'min_samples_split': [2, 3, 4, 8],
                    'min_samples_leaf': [1, 3, 5], 
                    'max_features': [2, 5]
                     },
              
              'KNN': {
                        'n_neighbors': [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                        'weights': ['uniform', 'distance'],
                        'metric': ['euclidean', 'manhattan']
                    },
              
              'GP': {'kernel': [1.0 * RBF(length_scale=1.0),  
                                1.0 * RBF(length_scale=1.5), 
                                1.0 * RBF(length_scale=2.0),
                                1.0 * RBF(length_scale=2.5),
                                1.0 * RBF(length_scale=3.0),
                                1.0 * RBF(length_scale=3.5),
                                1.0 * Matern(length_scale=1.0, nu=1.5),
                                1.0 * Matern(length_scale=1.5, nu=1.5),
                                1.0 * Matern(length_scale=1.5, nu=2.0),
                                1.0 * Matern(length_scale=2, nu=2.0)],
                     'alpha': [0.001, 0.01, 0.1, 1, 1.5, 2, 2.5, 3.0, 3.5]},
                     
                'xgboost':  {
                        'n_estimators': [100, 200, 300],
                        'max_depth':    [3, 5, 7, 10],
                        'gamma': [0, 1, 5],
                        'reg_alpha': [0, 1],  # L1 regularization
                        'reg_lambda': [1, 2]  # L2 regularization
                    },

               'Linear': {},
            }

# Finetunin after feature selection
# params = {
#     'SVM': {'C':100, 
#             'gamma':0.01, 
#             'kernel':'rbf'
#             },
#     'RF': {'n_estimators': 120,
#             'max_depth': 15,
#             'min_samples_split': 16,
#             'min_samples_leaf': 6 
#             },
#     'KNN': {'metric': 'manhattan', 
#             'n_neighbors': 5, 
#             'weights': 'distance'
#             },
#     'GP': {
#             'kernel': 1**2 * RBF(length_scale=2.5),
#             'alpha': 2
#     },
#    'Ridge': {'alpha':0.0005},
#    'Lasso': {'alpha':1e-05}          
# }


#### Hyperparamters Without Norm ############
# Initial params

# params = {
#     'SVM': {'C':200, 
#             'gamma':0.001, 
#             'kernel':'rbf'
#             },
#     'RF': {'n_estimators': 200,
#             'max_depth': 30,
#             'min_samples_split': 2,
#             'min_samples_leaf': 4 
#             },
#     'KNN': {'metric': 'manhattan', 
#             'n_neighbors': 3, 
#             'weights': 'distance'
#             },
#     'GP': {
#             'kernel': 1**2 * Matern(length_scale=1, nu=1.5)
#     },

#     'Ridge':{'alpha':0.01},
#     'Lasso': {'alpha': 0.001}
# }

# Fine tuned
# params = {
#     'SVM': {'C':100, 
#             'gamma':0.005, 
#             'kernel':'rbf'
#             },
#     'RF': {'n_estimators': 80,
#             'max_depth': 15,
#             'min_samples_split': 16,
#             'min_samples_leaf': 2
#             },
#     'KNN': {'metric': 'manhattan', 
#             'n_neighbors': 3, 
#             'weights': 'distance'
#             },
#     'GP': {
#             'kernel': 1**2 * Matern(length_scale=1, nu=1.5)
#     },

#     'Ridge':{'alpha':0.0001},
#     'Lasso': {'alpha':3e-05}
# }