import numpy as np
from sklearn.model_selection import KFold
import Data

#######################
# X_train, X_test; Y_train, Y_test; All set to be ndarrays with the right dimensions.
New_X = KFold(n_splits=10, shuffle=True, random_state=16)
X_train_list = []
X_test_list = []
Y_train_list = []
Y_test_list = []
for train_index, test_index in New_X.split(Data.X.values):
    X_train_list.append(Data.X.values[train_index])
    X_test_list.append(Data.X.values[test_index])
    Y_train_list.append(Data.Y_real.values[train_index])
    Y_test_list.append(Data.Y_real.values[test_index])

# X_train[i] means the X element of the training examples in the ith split(10 in total)
X_train = np.array(X_train_list, dtype=object)
X_test = np.array(X_test_list, dtype=object)
Y_train = np.array(Y_train_list, dtype=object)
Y_test = np.array(Y_test_list, dtype=object)
