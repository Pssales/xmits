import model
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score


def get_best_threshold(y_true, y_hat):

    threshold, max_auc = None, None
    
    for t in np.arange(.1, 1., .1):
        cdm = [k for k in y_hat]
        tmp = np.array(cdm, dtype=np.float)
        # print(tmp)      

        tmp[tmp >= t] = 1.
        tmp[tmp < t] = 0.
        auc = roc_auc_score(y_true, tmp)

        if max_auc is None or max_auc < auc:
            threshold, max_auc = t, auc

    return threshold

def xgb(x, y, x_test):

    xgb = model.ExtremeGradientBoosting()

    xgb.optimize(x, y, verbose_eval=None)

    y_hat = xgb.predict(x_test)

    return y_hat, get_best_threshold(y, xgb.predict(x))


def extratree(x, y, x_test):

    result = None

    n_estimators = 2000
    
    max_features = 5

    if np.size(x, 1) > 5:

        tree = ExtraTreesClassifier(n_estimators=n_estimators, max_features=max_features)

        tree.fit(x, y)

        return tree.predict(x_test), get_best_threshold(y, tree.predict_proba(x))

    return result, None


def logreg(x, y, x_test):

    logreg = LogisticRegression()

    scaler = preprocessing.StandardScaler()

    x = scaler.fit_transform(x)

    x_test = scaler.transform(x_test)

    logreg.fit(x, y)

    return logreg.predict_proba(x_test)[:,1], get_best_threshold(y, logreg.predict_proba(x)[:,1])


def mlp(x, y, x_test, hidden_units=128):

    mlp = MLPClassifier(hidden_layer_sizes=(hidden_units, hidden_units, hidden_units), max_iter=1000)

    scaler = preprocessing.StandardScaler()

    x = scaler.fit_transform(x)

    x_test = scaler.transform(x_test)

    mlp.fit(x, y)

    return mlp._predict(x_test)[:,0], get_best_threshold(y, mlp._predict(x)[:,0])
