DATASET_PATH = '/Users/sangam/Desktop/Epilepsey/Code/vgramreg/ML1_ML2'
OUTPUT_PATH  = 'Outputs'

name_conversion = {'Linear':'multivariate', 'std':'univariate, std(S)', 'mean':'univariate, mean(S)', 'peak area':'univariate, area(S)', \
                        'dS_dV_area':'univariate, area(dS/dV)', 'dS_dV_max_peak':'univariate, max(dS/dV)', 'dS_dV_min_peak':'univariate, min(dS/dV)',\
                    'peak height':'univariate, max(S)', 'dS_dV_peak_diff':'univariate, max(dS/dV) - min(dS/dV)', \
                    'peak V':'univariate, V_at_max(S)', 'dS_dV_max_V':'univariate, V_at_max(dS/dV)', 'dS_dV_min_V':'univariate, V_at_min(dS/dV)',\
                    'vcenter':'vcenter', 'peak curvature': 'peak curvature', 'KNN':'KNN', 'RF':'Random Forest', 'GP':'Gaussian Process'}
    
reverse_name_conversion = {}

for i in name_conversion:
    reverse_name_conversion[name_conversion[i]] = i
    
model_name_conversion = {'Linear':'Linear', 'KNN':'KNN', 'RF':'Random Forest', 'GP':'Gaussian Process', 'multivariate':'multivariate'}

models_features_r2 = {
                      'univariate, mean(S)': ['univariate, mean(S)'], \
                      'univariate, area(S)': ['univariate, area(S)'], \
                      'univariate, std(S)':['univariate, std(S)'], \
                      'univariate, max(S)':['univariate, max(S)'], \
                      'univariate, V_at_max(dS/dV)':['univariate, V_at_max(dS/dV)'], \
                      'univariate, V_at_max(S)':['univariate, V_at_max(S)'], \
                      'vcenter':['vcenter'], \
                      'univariate, V_at_min(dS/dV)':['univariate, V_at_min(dS/dV)'], \
                      'univariate, min(dS/dV)':['univariate, min(dS/dV)'], \
                      'univariate, max(dS/dV)':['univariate, max(dS/dV)'], \
                      'univariate, max(dS/dV) - min(dS/dV)':['univariate, max(dS/dV) - min(dS/dV)'], \
                      'peak curvature':['peak curvature'], \
                      'univariate, area(dS/dV)':['univariate, area(dS/dV)']
                         }

models_features_per = models_features_r2.copy()


paired_test = [('Linear', 'univariate, std(S)'), \
               ('Linear', 'KNN'), \
               ('Linear', 'RF'), \
               ('Linear', 'GP')
              ]