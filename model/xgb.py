import xgboost as xgb
import random


class ExtremeGradientBoosting(object):

    def __init__(self):

        self.__model = None

        self.__params = None

    def optimize(self, train_x, train_y, valid_x=None,
                 valid_y=None, eta=0.02, max_depth=4, subsample=1., colsample_bytree=1.,
                 objective='binary:logistic', max_iterations=5000, eval_metric='auc',
                 silent=True, seed=random.randint(0, 1000000), verbose_eval=5):

        self.__params = {'eta': eta,
                         'max_depth': max_depth,
                         'subsample': subsample,
                         'colsample_bytree': colsample_bytree,
                         'objective': objective,
                         'eval_metric': eval_metric,
                         'seed': seed,
                         'silent': silent}

        d_train = xgb.DMatrix(train_x, train_y)

        if valid_x is None:
            valid_x = train_x
            valid_y = train_y

        watchlist = [(xgb.DMatrix(train_x, train_y), 'train'), (xgb.DMatrix(valid_x, valid_y), 'valid')]

        self.__model = xgb.train(self.__params, d_train, max_iterations, watchlist, verbose_eval=verbose_eval, early_stopping_rounds=100)

    def predict(self, x):

        return self.__model.predict(xgb.DMatrix(x))
