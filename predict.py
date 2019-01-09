from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from feature_extraction import *
import pandas as pd
import numpy as np
import model

res = list()
def feature_extractor(sentence,diagram):
    results = feature.extract_key_words(sentence,diagram)
    array = []
    for i in range(0, 3):
        res.append(len(results[i]['adverb']))
        res.append(len(results[i]['adjective']))
        res.append(len(results[i]['article']))
        res.append(len(results[i]['noun']))
        res.append(len(results[i]['verb']))
        res.append(len(results[i]['preposition']))
        res.append(len(results[i]['pronoun']))

    return res

dataset = pd.read_csv('files/usecase_and_diagram.csv', sep=',')

del dataset['diagrama']
del dataset['scenario']
del dataset['Requisito']

x = dataset.iloc[:,:-7]

y = dataset['resposta'].astype(int)

teste = feature_extractor("T CSGB ST=2 SST=1 endereço TCendereço [accept]TM VC [endereço inválido]TM RE [endereço válido]traduzendereço sinal Encaminha TM cmd pusosinal ","Pacote TC_CTL.TC_MudancaMododeOperacao recebido e mudança de modo é inválida. Gerar e enviar TM_RE")

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(x, y)                         
MLPClassifier(hidden_layer_sizes=(128, 128, 128), max_iter=1000)

print(clf.predict([teste]))