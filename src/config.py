from sklearn.gaussian_process.kernels import Matern, RBF

DATASET_PATH = '/Users/sangam/Desktop/Epilepsey/Code/vgramreg/ML1_ML2'
OUTPUT_PATH  = 'Outputs'

name_conversion = {'Linear':'multivariate', 'std':'univariate, std(S)', 'mean':'univariate, mean(S)', 'peak area':'univariate, area(S)', \
                        'dS_dV_area':'univariate, area(dS/dV)', 'dS_dV_max_peak':'univariate, max(dS/dV)', 'dS_dV_min_peak':'univariate, min(dS/dV)',\
                    'peak height':'univariate, max(S)', 'dS_dV_peak_diff':'univariate, max(dS/dV) - min(dS/dV)', \
                    'peak V':'univariate, V_max(S)', 'dS_dV_max_V':'univariate, V_max(dS/dV)', 'dS_dV_min_V':'univariate, V_min(dS/dV)',\
                    'vcenter':'vcenter', 'peak curvature': 'peak curvature', 'KNN':'KNN', 'RF':'Random Forest', 'GP':'Gaussian Process'}
    
reverse_name_conversion = {}

for i in name_conversion:
    reverse_name_conversion[name_conversion[i]] = i
    
model_name_conversion = {'Linear':'Linear', 'KNN':'KNN', 'RF':'Random Forest', 'GP':'Gaussian Process', 'multivariate':'multivariate', 'SVM':'SVM', 'Lasso':'Lasso', 'Ridge':'Ridge'}

models_features_r2 = {
                      'univariate, mean(S)': ['univariate, mean(S)'], \
                      'univariate, area(S)': ['univariate, area(S)'], \
                      'univariate, std(S)':['univariate, std(S)'], \
                      'univariate, max(S)':['univariate, max(S)'], \
                      'univariate, V_max(dS/dV)':['univariate, V_max(dS/dV)'], \
                      'univariate, V_max(S)':['univariate, V_max(S)'], \
                      'vcenter':['vcenter'], \
                      'univariate, V_min(dS/dV)':['univariate, V_min(dS/dV)'], \
                      'univariate, min(dS/dV)':['univariate, min(dS/dV)'], \
                      'univariate, max(dS/dV)':['univariate, max(dS/dV)'], \
                      'univariate, max(dS/dV) - min(dS/dV)':['univariate, max(dS/dV) - min(dS/dV)'], \
                      'peak curvature':['peak curvature'], \
                      'univariate, area(dS/dV)':['univariate, area(dS/dV)'], \
                #       'Linear':['univariate, std(S)', 'univariate, V_max(S)', 'univariate, min(dS/dV)', 'univariate, V_max(dS/dV)', 'peak curvature'],
                #       'KNN':   ['univariate, area(S)', 'univariate, V_max(S)', 'univariate, min(dS/dV)', 'univariate, std(S)'],
                #       'RF':    ['univariate, std(S)', 'peak curvature', 'univariate, min(dS/dV)', 'univariate, max(dS/dV) - min(dS/dV)'],
                #       'GP':    ['univariate, std(S)', 'univariate, V_max(dS/dV)', 'univariate, V_max(S)'],
                #       'SVM':   ['univariate, std(S)', 'peak curvature', 'univariate, V_max(dS/dV)', 'univariate, max(dS/dV)', 'univariate, V_max(S)'],
                #       'Lasso': ['univariate, std(S)', 'univariate, V_max(S)', 'univariate, min(dS/dV)', 'univariate, V_max(dS/dV)', 'peak curvature'],
                #       'Ridge': ['univariate, std(S)', 'univariate, V_max(S)', 'univariate, min(dS/dV)', 'univariate, V_max(dS/dV)', 'peak curvature']
                'Linear': ['univariate, std(S)', 'univariate, V_max(S)', 'univariate, min(dS/dV)', 'univariate, V_max(dS/dV)', 'peak curvature'],
                'KNN': ['univariate, max(S)', 'univariate, std(S)', 'univariate, mean(S)', 'univariate, V_max(S)'],
                'RF': ['univariate, std(S)', 'univariate, max(dS/dV) - min(dS/dV)'],
                'GP': ['univariate, std(S)'],
                'SVM': ['univariate, area(S)', 'univariate, V_max(dS/dV)', 'univariate, max(dS/dV)', 'univariate, V_max(S)', 'vcenter', 'univariate, area(dS/dV)', 'univariate, mean(S)'],
                'Lasso': ['univariate, std(S)', 'univariate, V_max(S)', 'univariate, min(dS/dV)', 'univariate, V_max(dS/dV)', 'peak curvature'],
                'Ridge': ['univariate, std(S)', 'univariate, V_max(S)', 'univariate, min(dS/dV)', 'univariate, V_max(dS/dV)', 'peak curvature', 'univariate, max(dS/dV)', 'univariate, max(dS/dV) - min(dS/dV)'],
                    
                }

models_features_per = models_features_r2.copy()
models_features_per.update({
                #       'Linear': ['univariate, std(S)', 'vcenter', 'univariate, area(dS/dV)', 'peak curvature', 'univariate, V_max(dS/dV)', 'univariate, V_min(dS/dV)', 'univariate, max(dS/dV) - min(dS/dV)', 'univariate, max(S)'],
                #       'KNN':   ['univariate, std(S)', 'univariate, area(dS/dV)'],
                #       'RF':    ['univariate, std(S)', 'univariate, max(dS/dV) - min(dS/dV)', 'peak curvature'],
                #       'GP':    ['univariate, std(S)', 'univariate, V_max(dS/dV)', 'univariate, V_max(S)'],
                #       'SVM':   ['univariate, area(dS/dV)', 'univariate, max(dS/dV) - min(dS/dV)', 'univariate, V_max(dS/dV)', 'univariate, V_max(S)', 'univariate, std(S)', 'univariate, max(S)', 'univariate, min(dS/dV)', 'univariate, area(S)', 'univariate, max(dS/dV)', 'univariate, mean(S)'],
                #       'Lasso': ['univariate, std(S)', 'vcenter', 'univariate, area(dS/dV)', 'univariate, V_max(S)', 'peak curvature', 'univariate, V_max(dS/dV)', 'univariate, V_min(dS/dV)', 'univariate, max(dS/dV)', 'univariate, max(S)'],
                #       'Ridge': ['univariate, std(S)', 'vcenter', 'univariate, area(dS/dV)', 'peak curvature', 'univariate, V_max(dS/dV)', 'univariate, V_min(dS/dV)', 'univariate, max(dS/dV) - min(dS/dV)', 'univariate, V_max(S)', 'univariate, max(S)']    
                'Linear': ['univariate, std(S)', 'vcenter', 'univariate, area(dS/dV)', 'peak curvature', 'univariate, V_max(dS/dV)', 'univariate, V_min(dS/dV)', 'univariate, max(dS/dV) - min(dS/dV)', 'univariate, max(S)'],
                'KNN': ['univariate, max(dS/dV) - min(dS/dV)', 'univariate, area(S)', 'univariate, min(dS/dV)', 'univariate, V_min(dS/dV)', 'univariate, V_max(dS/dV)', 'univariate, V_max(S)', 'vcenter', 'univariate, max(S)'],
                'RF': ['univariate, std(S)', 'univariate, min(dS/dV)'],
                'GP': ['univariate, std(S)', 'univariate, V_max(dS/dV)', 'univariate, V_max(S)', 'univariate, mean(S)'],
                'SVM': ['univariate, max(dS/dV) - min(dS/dV)', 'univariate, area(S)', 'univariate, V_max(dS/dV)', 'univariate, max(dS/dV)', 'univariate, V_max(S)', 'univariate, V_min(dS/dV)', 'univariate, min(dS/dV)', 'vcenter', 'univariate, area(dS/dV)', 'univariate, max(S)', 'univariate, std(S)', 'univariate, mean(S)'],
                'Lasso': ['univariate, std(S)', 'vcenter', 'univariate, area(dS/dV)', 'univariate, V_max(S)', 'peak curvature', 'univariate, V_max(dS/dV)', 'univariate, V_min(dS/dV)', 'univariate, max(dS/dV) - min(dS/dV)'],
                'Ridge': ['univariate, std(S)', 'vcenter', 'univariate, V_max(S)', 'univariate, V_max(dS/dV)', 'univariate, min(dS/dV)'],
                })

paired_test = [('Linear', 'univariate, std(S)'), 
               ('Linear', 'univariate, max(dS/dV)'),
               ('univariate, max(dS/dV)', 'KNN'),
               ('univariate, max(dS/dV)', 'RF'),
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
    'SVM': {'C':100, 
            'gamma':0.01, 
            'kernel':'rbf'
            },
    'RF': {'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 2 
            },
    'KNN': {'metric': 'manhattan', 
            'n_neighbors': 7, 
            'weights': 'uniform'
            },
   'GP': {
            'kernel': 1**2 * RBF(length_scale=2.5),
            'alpha': 2
    },
    'Ridge': {'alpha':0.01},
    'Lasso': {'alpha':0.001}
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