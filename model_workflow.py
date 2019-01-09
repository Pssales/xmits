from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import model
import matplotlib.pyplot as plt


dataset = pd.read_csv('files/usecase_and_diagram.csv', sep=',')

del dataset['diagrama']
del dataset['scenario']

x = dataset.iloc[:,:-7]

y = dataset['resposta'].astype(int)

# print(dataset.info())

fold = KFold(n_splits=15)

auc = {}

for func in [model.xgb,model.mlp,model.logreg]:

    auc[func.__name__] = []

    for train_index, test_index in fold.split(x):

        x_train, y_train = x.iloc[train_index,:], y.as_matrix()[train_index]

        x_test, y_test = x.iloc[test_index, :], y.as_matrix()[test_index]

        y_hat, t = func(x_train, y_train, x_test)

        #y_hat[y_hat >= .5] = 1
        #y_hat[y_hat < .5] = 0

        auc_tmp = [roc_auc_score(y_test, y_hat)]

        auc[func.__name__] += [auc_tmp]
teste= {};
for key in auc:
    teste[key] = auc[key]
    print(key, ':', np.mean(auc[key]), np.std(auc[key]))


