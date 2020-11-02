import KFold
import Data
import numpy as np

#######################
# Data of 10 * m * n ( X_train[0] is an ndarray of 3759 * 8 )
X_train = KFold.X_train
Y_train = KFold.Y_train
X_test = KFold.X_test
Y_test = KFold.Y_test

#########################
# Method 1: Min-Max Normalization
X_min_max_train = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
X_min_max_test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(10):
    X_Min = Data.X.values.min(axis=0)
    X_Max = Data.X.values.max(axis=0)
    X_Diff = X_Max - X_Min
    X_min_max_train[i] = (X_train[i] - X_Min) / X_Diff
    X_min_max_test[i] = (X_test[i] - X_Min) / X_Diff


# Method 2: Mean Normalization
X_mean_train = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
X_mean_test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(10):
    X_Min = Data.X.values.min(axis=0)
    X_Max = Data.X.values.max(axis=0)
    X_Diff = X_Max - X_Min
    X_Mean = Data.X.values.mean(axis=0)
    X_mean_train[i] = (X_train[i] - X_Mean) / X_Diff
    X_mean_test[i] = (X_test[i] - X_Mean) / X_Diff


# Method3: Standardization
X_Std_train = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
X_Std_test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in range(10):
    X_Mean = Data.X.values.mean(axis=0)
    X_Std = np.std(np.array(Data.X.values, dtype=np.float64), axis=0)
    X_Std_train[i] = (X_train[i] - X_Mean) / X_Std
    X_Std_test[i] = (X_test[i] - X_Mean) / X_Std
