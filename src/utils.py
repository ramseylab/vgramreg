import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

def find_adj_score(N: int, P: int, R_2: float) -> float:
    return (1 - (1 - R_2)*(N - 1)/(N - P - 1))

def per_error(y_test:pd.Series, y_pred:np.array, y_LOD:float)->float:
   
    mask           = (y_test != 0)    # Non Zero Concentration
    zero_mask      = ~(mask)          # Zero Concentration

    y_pred         = np.maximum(y_pred, 0.0)

    # Only for non zero concentration
    non_zero_per_error = np.abs(y_test[mask] - y_pred[mask])/(0.5*(y_test[mask] + y_pred[mask]))
   
    # zero concentration
    zero_per_error     = np.abs(y_test[zero_mask] - y_pred[zero_mask]) / y_LOD

    assert not(np.isnan(zero_per_error).any())
    assert not(np.isnan(non_zero_per_error).any())

    per_error         = np.concatenate((non_zero_per_error, zero_per_error))
    per_error         = np.mean(per_error) * 100

    return per_error

def calculate_y_LOD(X_train, y_train):
    model_yLOD = LinearRegression()
    model_yLOD.fit(X_train[['univariate, std(S)']], y_train)          # Selecting standard deviation of sample as a feature
    
    S  = model_yLOD.coef_[0]                                                    # Slope of fitting line y=Sx + c 
    SD = X_train['univariate, std(S)'][(y_train==0).to_numpy()].std() # Standard deviation of S blank

    return 2.636369 * S * SD # We got the constant value from the -qt(0.01/2, 83) there number of blanks = 84 and we are using k-1 degree 84 -1 = 83

