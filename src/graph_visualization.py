import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from typing import Tuple

def pad_values(feature_:list, scores:list, max_val:int) -> Tuple[list, list]:
    pad_value = max_val - len(feature_)
    feature_  = feature_ + ['']*pad_value
    scores    = scores   + [0]*pad_value

    return feature_, scores

def feature_selection_tabularize(feature_scores: dict) -> pd.DataFrame:

    ind_header_name = {1:'Univariate Features', 2:'Bivariate Features', 3:'Trivariate Features', 4:'Quad', 5:'Quin', \
                      6: 'Hexvariate Features', 7:'Heptavariate Features', 8:'Octavaraite Features', 9:'novemvariate Features', 10:"Ten", 11:'',12:'', 13:''}
    
    num_features = len(feature_scores[0])
    df           = pd.DataFrame()
        
    for ind, all_feature in enumerate(feature_scores):
        
        feature_, scores   = pad_values(list(all_feature.keys()), list(all_feature.values()), num_features)
        scores             = [round(float(i[0]), 4) if (type(i)==np.ndarray)  else round(float(i), 4) for i in scores]

        
        df[ind_header_name[ind + 1]]  = feature_
        df[f'Scores_{ind+1}']         = scores

    return df

def visualize_highest_score_feature_selection(all_dataset_feature_score: dict, 
                                              path_name:str, 
                                              model_name_conversion: dict, 
                                              r2_score=True, 
                                              adj_score=False,
                                              only_one_multivariate=True, 
                                              legends=False, 
                                              extra_symbol=None) -> pd.DataFrame:

    os.makedirs(os.path.dirname(path_name), exist_ok=True)
    plt.figure(figsize=(35, 15))

    dict_ = {}

    dict_['Models']   = []
    dict_['Scores']   = []
    dict_['features'] = []

    all_dataset_feature_score = {'Linear':all_dataset_feature_score['Linear']} if only_one_multivariate else all_dataset_feature_score
    
    for dataset_name in all_dataset_feature_score:
        best_score       = 0.0 if r2_score else 100.0
        selected_feature = None
        fontsize         = 14
        
        for ind, all_feature in enumerate(all_dataset_feature_score[dataset_name]):
            feature_   = list(all_feature.keys())
            scores     = [i[1 if adj_score else 0] if (type(i)==np.ndarray) else i for i in list(all_feature.values())]

            if r2_score: score, ind_  = np.max(scores), np.argmax(scores)
            else: score, ind_  = np.min(scores), np.argmin(scores)
                
           
            if r2_score:
                if score > best_score:
                    selected_feature = feature_[ind_]
                    best_score       = score
            else:
                 if score < best_score:
                    selected_feature = feature_[ind_]
                    best_score = score

            # print(dataset_name, ind)
            if ('Linear' == dataset_name) and (ind ==0) and (only_one_multivariate):
                argsort_ = np.argsort(scores)[-4:][::-1] if r2_score else np.argsort(scores)[:4]
                for i in argsort_:
                    dict_['Models'].append(f'{eval(feature_[i])[0]}')
                    dict_['Scores'].append(scores[i])
                    dict_['features'].append(feature_[i])

        dataset_name = model_name_conversion[dataset_name]

        if only_one_multivariate==True: dataset_name = 'multivariate'
            
        dict_['Models'].append(dataset_name)
        dict_['Scores'].append(best_score)
        dict_['features'].append(selected_feature)

    df = pd.DataFrame.from_dict(dict_)

    if not(r2_score): df = df.sort_values(by='Scores')
    else: df = df.sort_values(by='Scores', ascending=False)

    ax = df[['Models', 'Scores']].plot(x='Models', y=['Scores'], kind='bar', legend=False, color='0.7', edgecolor='black', fontsize=fontsize)

    symbols      = {'multivariate':'o', 'univariate, std(S)':'^', 'univariate, mean(S)':'x',\
                 'univariate, area(S)':'v', \
                 'univariate, area(dS/dV)':'D', \
                 'univariate, max(S)':'*', \
                 'KNN': 'o',\
                 'Random Forest': '^',\
                 'Gaussian Process': 'v',
                 'univariate, max(dS/dV)': 'v',\
                 'Linear': 'x',
                 'SVM':'*',
                 'Ridge':'+',
                 'Lasso':'s'}
    
    decimal_prec = 3 if r2_score else 1

    for i, p in enumerate(ax.patches):
        ax.annotate(str(round(p.get_height(), decimal_prec)), (p.get_x() + p.get_width() / 2., 0.3 * p.get_height()),
            ha='center', va='center', xytext=(0, 10), textcoords='offset points',  rotation='vertical', fontsize=fontsize)

        color = 'red' if i == 0 else 'black' 
        ax.plot(p.get_x() + p.get_width() / 2, p.get_height(), marker=symbols[df['Models'].to_list()[i]], \
                color=color, markersize=10)

    if extra_symbol != None:
        ax.plot(-1, -1, marker=extra_symbol,color=color, markersize=10)
    
    ylabel = r'$R^2$' if r2_score else 'Error (%)'

    # Set y limit for R2 and percent error
    set_y_lim = 1.0 if r2_score else 70.0
    plt.ylim(top=set_y_lim)
    plt.ylabel(ylabel, fontsize=fontsize+2)
    plt.xlabel('')

    if r2_score: 
        ax.set_xticklabels([])
        plt.xlabel('Features', fontsize=14)
    
    if legends:
        leg = plt.legend(df['Models'].values.tolist(), loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

        for line in leg.get_lines():
            line.set_color('black')
        
    ax.set_xticklabels([])

    if (r2_score and adj_score):  path_name = path_name.split('.')[0]+"_Adj.png"

    plt.xlabel('Features', fontsize=14)
    plt.savefig(f'{path_name}', dpi=300, bbox_inches='tight')
    plt.clf()

    return df

def visualization_testing_dataset(dict_:dict, 
                                  path_name:str, 
                                  model_name_conversion:dict,
                                  only_one_multivariate=True,
                                  r2_score=True, 
                                  adj_score = False,
                                  legends=False) -> None:
    fontsize = 14
    df       = pd.DataFrame.from_dict(dict_)

    if not(r2_score): df = df.sort_values(by='Scores')
    else: 
        df['Scores'] = df['Scores'].apply(lambda x: x[1]) if adj_score else df['Scores'].apply(lambda x: x[0])
        df = df.sort_values(by='Scores', ascending=False)

    if not (only_one_multivariate): df['Models'] = df['Models'].apply(lambda x: model_name_conversion[x] if (x in model_name_conversion) else x )
    ax = df[['Models', 'Scores']].plot(x='Models', y=['Scores'], kind='bar', legend=False, color='0.7', edgecolor='black', fontsize=fontsize)

    symbols      = {'multivariate':'o', 'univariate, std(S)':'1', 'univariate, mean(S)':'x',\
                 'univariate, area(S)':'2', \
                 'univariate, area(dS/dV)':'D', \
                 'univariate, max(S)':'*', \
                 'KNN': 'o',\
                 'Random Forest': '^',\
                 'RF': '^',\
                 'Gaussian Process': 'v',\
                 'univariate, max(dS/dV)': '3',\
                 'GP': 'v',\
                 'Linear': 'x',
                 'SVM':'*',
                 'Ridge':'+',
                 'Lasso':'s'}
    
    decimal_prec = 3 if r2_score else 1

    for i, p in enumerate(ax.patches):
        ax.annotate(str(round(p.get_height(), decimal_prec)), (p.get_x() + p.get_width() / 2., 0.3 * p.get_height()),
            ha='center', va='center', xytext=(0, 10), textcoords='offset points',  rotation='vertical', fontsize=fontsize)

        if legends or r2_score:
            color = 'red' if i == 0 else 'black' 
            ax.plot(p.get_x() + p.get_width() / 2, p.get_height(), marker=symbols[df['Models'].to_list()[i]], \
                    color=color, markersize=10)

    ylabel    = r'$R^2$' if r2_score else 'Error (%)'
    set_y_lim = 1.0 if r2_score else 70.0
    
    plt.ylim(top=set_y_lim)
    plt.ylabel(ylabel, fontsize=fontsize+2)
    plt.xlabel('')
    ax.set_xticklabels([])

    if r2_score: 
        ax.set_xticklabels([])
        plt.xlabel('Features', fontsize=14)
    
    if legends:
        leg = plt.legend(df['Models'].values, loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

        for line in leg.get_lines():
            line.set_color('black')

        ax.set_xticklabels([])
        plt.xlabel('Features', fontsize=14)
        
    if (r2_score and adj_score):  path_name = path_name.split('.')[0]+"_Adj.png"  

    plt.savefig(f'{path_name}', dpi=300, bbox_inches='tight')
    plt.clf()