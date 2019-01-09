from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import model


dataset = pd.read_csv('files/usecase_and_diagram.csv', sep=',')

del dataset['diagrama']
del dataset['scenario']

x = dataset.iloc[:,:-7]

y = dataset['resposta'].astype(int)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(x, y)                         
MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(5, 2),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=200, momentum=0.9,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)


print(clf.predict([[0,1,0,53,1,0,0,2,5,7,27,11,4,0,0,0,0,0,0,0,0,4]]))