from sklearn.ensemble import RandomForestRegressor
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
import joblib
import pandas as pd
from skopt import BayesSearchCV
import warnings

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from skopt.space import Real, Integer, Categorical
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from data_utils.utils import Data_Pro

class ModelBase:
    def __init__(self, X_train=None, 
                 y_train=None, 
                 X_test=None, 
                 y_test=None, 
                 kf=None, 
                 model_save_file=None, 
                 target=None, 
                 method=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.kf = kf
        self.model_save_file = model_save_file
        self.target = target
        self.method = method
        self.model = None
        self.param_grids = {
            "cat":{
                'n_estimators': Integer(100, 300),
                'learning_rate': Real(0.01, 0.2),
                'max_depth': Integer(3, 10),
                'subsample': Real(0.5, 1.0),
                'colsample_bylevel': Real(0.5, 1.0),
                'reg_lambda': Real(0.01, 5)
            },
            
            'rf': {
                'n_estimators': Integer(100, 500),
                'max_depth': Integer(3, 20),
                'min_samples_split': Integer(2, 20)
            },
            'xgb': {
                'n_estimators': Integer(100, 500),
                'learning_rate': Real(0.01, 0.5),
                'max_depth': Integer(3, 20),
                'gamma': Real(0, 1),
                'min_child_weight': Integer(1, 10),
                'subsample': Real(0.5, 1.0),
                'colsample_bytree': Real(0.5, 1.0),
                'tree_method': Categorical(['auto', 'exact', 'approx', 'hist'])
            },
            'lgb': {
                'n_estimators': Integer(100, 500),
                'learning_rate': Real(0.01, 0.5),
                'max_depth': Integer(3, 10),
                'num_leaves': Integer(10, 50),
                'subsample': Real(0.5, 1.0),
                'colsample_bytree': Real(0.5, 1.0),
            },
            'Decision Tree': {
                'max_depth': Integer(3, 20),
                'min_samples_split': Integer(2, 20)
            },
             "mlp": {
                'hidden_layer_sizes': Integer(50, 200),
                'activation': Categorical(['relu', 'tanh']),
                'solver': Categorical(['adam', 'lbfgs']),
                'alpha': Real(0.0001, 0.1),
                'learning_rate': Categorical(['constant', 'adaptive']),
                'max_iter': Integer(200, 1000)
            },
            "svr": {
                'C': Real(0.1, 10),
                'epsilon': Real(0.01, 0.1),
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']  
            },
                "gsr": {
                    'alpha': Real(1e-10, 1e-1, prior='log-uniform'),
                    'n_restarts_optimizer': Integer(0, 10),
                    'normalize_y': [True, False]
                },
            "ridgelr": {
                'alpha': Real(0.01, 10),
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            },
            'lr':{
                
            }
            
    }

    def isBayesian(self):
        # 确保 X_train 和 y_train 中没有 NaN 或 inf 值
        self.X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.y_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.X_train.dropna(inplace=True)
        self.y_train.dropna(inplace=True)
        
        # 检查数据类型和范围
        if not np.isfinite(self.X_train.values).all():
            raise ValueError("X_train contains non-finite values.")
        if not np.isfinite(self.y_train).all():
            raise ValueError("y_train contains non-finite values.")
        if np.any(np.abs(self.X_train.values) > np.finfo(np.float64).max):
            raise ValueError("X_train contains values too large for dtype('float64').")
        if np.any(np.abs(self.y_train) > np.finfo(np.float64).max):
            raise ValueError("y_train contains values too large for dtype('float64').")
        if self.param_grids[self.method]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                bayes_search = BayesSearchCV(estimator=self.model,
                                            search_spaces=self.param_grids[self.method],
                                            n_iter=50,
                                            cv=self.kf,
                                            scoring='r2',
                                            n_jobs=-1,
                                            random_state=42,
                                            verbose=False)
                # 预处理数据
                self.X_train = Data_Pro.preprocess_data(self.X_train)
                bayes_search.fit(self.X_train, self.y_train)  # Use y_combined_train
                
            best_model = bayes_search.best_estimator_
            self.model = best_model
        else:
            # 如果没有超参数，则直接训练
            self.model.fit(self.X_train, self.y_train)
            score = self.model.score(self.X_train, self.y_train)
            print(f"Score for {self.method}: ", score)

    def train(self):
        X_train, y_train, kf = self.X_train, self.y_train, self.kf

        preds = np.zeros(X_train.shape[0])

        logging.info('{} Training Begin--------------------------------------------------------------------------------------'.format(self.method))
        for i, (train_index, valid_index) in enumerate(kf.split(X_train)):
            x_tr, x_va = X_train.iloc[train_index], X_train.iloc[valid_index]
            y_tr, y_va = y_train.iloc[train_index], y_train.iloc[valid_index]

            self.model.fit(x_tr, y_tr)
            pred_valid = self.model.predict(x_va)
            preds[valid_index] = pred_valid

            logging.info('{}, {}'.format(x_tr.shape, y_tr.shape))
            logging.info('Train: The rmse of the {} is: {}'.format(self.method, metrics.mean_squared_error(y_va,pred_valid)))
            logging.info('Train: The mae of the {} is: {}'.format(self.method, metrics.mean_absolute_error(y_va,pred_valid)))
            logging.info('Train: The R2 of the {} is: {}'.format(self.method, metrics.r2_score(y_va,pred_valid)))
            x = list(range(0, len(y_va)))
            plt.figure(figsize=(40,10))
            plt.plot(x, y_va, 'g', label='ground-turth')
            plt.plot(x, pred_valid, 'r', label=f'{self.method} predict')
            plt.legend(loc='best')
            plt.title(f'{self.method}-yield')
            plt.xlabel('sample')
            plt.ylabel('Yield_lnRR')
            plt.savefig(f'{self.save_path}/plot_{i}.png')  # 保存图形为 plot_0.png, plot_1.png, ...
            plt.close()  # 关闭图形，以释放内存

    def test(self):
        X_test, y_test, model_save_file = self.X_test, self.y_test, self.model_save_file
        
        pred_test = self.model.predict(X_test)
        self.pred_test = pred_test

        self.mse = metrics.mean_squared_error(y_test, pred_test)
        self.mae = metrics.mean_absolute_error(y_test, pred_test)
        self.r2 = metrics.r2_score(y_test, pred_test)
        logging.info('Test: The rmse of the {} is: {}'.format(self.method, self.mse))
        logging.info('Test: The mae of the {} is: {}'.format(self.method, self.mae))
        logging.info('Test: The R2 of the {} is: {}'.format(self.method, self.r2))

        x = list(range(0, len(y_test)))
        plt.figure(figsize=(40,10))
        plt.plot(x[:100], y_test[:100], 'g', label='ground-turth')
        plt.plot(x[:100], pred_test[:100], 'r', label=f'{self.method} predict')
        plt.legend(loc='best')
        plt.title({self.target})
        plt.xlabel('sample')
        plt.ylabel('Yield_lnRR')
        plt.savefig(f'{self.save_path}/plot_test.png')
        plt.close()

        return self.mse, self.mae, self.r2, self.pred_test

    def save_result(self):
        joblib.dump(self.model, f'{self.save_path}/{self.method}_model.pkl') 
        df_tmp = pd.DataFrame({'pred': self.pred_test, 'true': self.y_test})
        df_tmp.to_csv(f'{self.save_path}/{self.method}_pred.csv', index=False)
        print(f'{self.__class__.__name__} save ok...')

    def get_important_analyse(self):
        pass