import pandas as pd
import numpy as np

from glob import glob

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple

from src.config import DATASET_PATH, OUTPUT_PATH

def find_concentration_distribution(y: pd.Series) -> int:
    all_labels = y.tolist()
    unique_    = set(all_labels)
    count = {}
    for i in unique_: count[i] = all_labels.count(i)
    
    return count
def create_correlation_matrix(X_correl:pd.DataFrame) -> None:

    # Remove the univariate from the column name
    X_correl.columns = [name.replace('univariate, ', '') for name in X_correl.columns.to_list()]

    # Calculate the correlation matrix
    correlation_matrix = X_correl.corr()

    # Create a mask for the upper triangle
    mask = np.tril(np.ones_like(correlation_matrix, dtype=bool))

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, mask=~mask, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 12.5}, square=True)
    plt.title('Correlation Matrix')
    plt.savefig(f'{OUTPUT_PATH}/feature_correlation_matrix.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
     

def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    datasets = sorted([f"{i}/extracted_features.xlsx" for i in glob(f'{DATASET_PATH}/*')])
   
    df  = [pd.read_excel(dataset) for dataset in datasets]
    df  = pd.concat(df)
    
    X   = df[["peak area", "peak curvature", "peak V", "vcenter", "PH", "signal_mean", "signal_std", \
                                "dS_dV_max_peak", "dS_dV_min_peak", "dS_dV_peak_diff", "dS_dV_max_V", "dS_dV_min_V", "dS_dV_area"]]
    y   = df['file'].apply(lambda x: int(x.split('_')[-2].replace('cbz','')))

    X.rename(columns={"PH": 'univariate, max(S)', 'signal_std':'univariate, std(S)', 'signal_mean':'univariate, mean(S)', 'peak area':'univariate, area(S)', \
                        'dS_dV_area':'univariate, area(dS/dV)', 'dS_dV_max_peak':'univariate, max(dS/dV)', 'dS_dV_min_peak':'univariate, min(dS/dV)',\
                    'dS_dV_peak_diff':'univariate, max(dS/dV) - min(dS/dV)', \
                    'peak V':'univariate, V_max(S)', 'dS_dV_max_V':'univariate, V_max(dS/dV)', 'dS_dV_min_V':'univariate, V_min(dS/dV)',\
        }, inplace = True)

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler to your data
    scaler.fit(X)

    # Transform the data
    X_normalized = scaler.transform(X)
    
    # Transform the data
    X_normalized = scaler.transform(X)

    # Convert the numpy array back to a DataFrame
    X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

    # Generate Feature Correlation heat map
    create_correlation_matrix(X_normalized.copy())

    # Split the total dataset into training (70%) and testing (30%) dataset
    X_train, X_test, y_train, y_test  = train_test_split(X_normalized, y, test_size=0.4, shuffle=True, random_state=20, stratify=y)

    print("######Data Distribution:#########")
    print("Training", find_concentration_distribution(y_train))
    print("Testing",  find_concentration_distribution(y_test))
    print("#################################")
    return X_train, X_test, y_train, y_test