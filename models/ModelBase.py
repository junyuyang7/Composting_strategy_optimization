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
import sys

from data_utils.utils import Data_Pro

warnings.filterwarnings("ignore")


class ModelBase:
    def __init__(self, X_train=None, 
                 y_train=None, 
                 X_test=None, 
                 y_test=None, 
                 kf=None, 
                 model_save_file=None, 
                 target=None, 
                 method=None,
                 num_col=None,
                 is_bayesian=True):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.kf = kf
        self.model_save_file = model_save_file
        self.target = target
        self.method = method
        self.model = None
        if num_col:
            self.save_path = f'{model_save_file}/{method}_{num_col}'
        else:
            self.save_path = f'{model_save_file}/{method}'
        if X_train is not None:
            self.features = X_train.columns
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
            # 检查 y_tr 是否只有一个唯一值
            unique, counts = np.unique(y_tr, return_counts=True)
            if len(unique) == 1:
                print(f"Skipping fold {i} as all train targets are equal.")
                continue

            if self.method == 'cat':
                self.model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], use_best_model=True,verbose=0)
            else:
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
            
        self.pred_train = self.model.predict(X_train)

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
        if self.X_test is not None:
            df_tmp = pd.DataFrame({'pred': self.pred_test, 'true': self.y_test})
            df_tmp.to_csv(f'{self.save_path}/{self.method}_pred_te.csv', index=False)
            print(f'{self.save_path}/{self.method}_pred_te.csv save ok' )
        df_tmp_tr = pd.DataFrame({'pred': self.pred_train, 'true': self.y_train})
        df_tmp_tr.to_csv(f'{self.save_path}/{self.method}_pred_tr.csv', index=False)
        print(f'{self.save_path}/{self.method}_pred_tr.csv save ok' )
        print(f'{self.__class__.__name__} save ok...')

    def get_important_analyse(self):
        pass

    # 获取mse增加值
    def get_mse_increase(self):
        # 基线MSE
        baseline_mse = metrics.mean_squared_error(self.y_train, self.model.predict(self.X_train))
        mse_increase = []
        for feature in self.features:
            X_permuted = self.X_train.copy()
            X_permuted[feature] = np.random.permutation(self.X_train[feature])
            permuted_mse = metrics.mean_squared_error(self.y_train, self.model.predict(X_permuted))
            mse_increase.append(permuted_mse - baseline_mse)
        self.mse_increase = mse_increase
        print(f'{sys._getframe().f_code.co_name} finish')

    # 获取节点纯度增加
    def get_pure_increase(self):
        if self.method == 'rf':
            impurity_decrease = np.mean([
                tree.feature_importances_ for tree in self.model.estimators_
            ], axis=0)
        elif self.method == 'xgb':
            booster = self.model.get_booster()
            importance_dict = booster.get_score(importance_type='gain')
            impurity_decrease = np.array([importance_dict.get(f, 0.0) for f in self.features])
        elif self.method == 'cat':
            impurity_decrease = self.model.get_feature_importance(type='PredictionValuesChange')
            # leaf_importance = self.model.calc_leaf_importance()
            # impurity_decrease = np.zeros(len(self.features))
            # for leaf_imp in leaf_importance:
            #     impurity_decrease += np.array([leaf_imp[f] for f in self.features])
        elif self.method == 'lgb':
            impurity_decrease = self.model.booster_.feature_importance(importance_type='gain')
            # booster = self.model.booster_
            # importance_dict = booster.feature_importance(importance_type='gain')
            # impurity_decrease = np.array([importance_dict[i] for i in range(len(self.features))])
        else:
            raise '这种方法没有节点纯度这个概念的~~'
        self.impurity_decrease = impurity_decrease
        print(f'{sys._getframe().f_code.co_name} finish')
        
    def get_p_value(self):
        from sklearn.inspection import permutation_importance
        from scipy.stats import ttest_ind

        perm_importance = permutation_importance(self.model, self.X_train, self.y_train, n_repeats=30, random_state=0)
        p_values = np.array([ttest_ind(perm_importance.importances[i], perm_importance.importances_mean).pvalue for i in range(len(self.features))])
        self.p_values = p_values
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_important_mark(self):
        top_threshold = np.percentile(self.impurity_decrease, 75)  # 设定为前25%的特征为重要特征
        top_features = self.impurity_decrease >= top_threshold
        self.top_features = top_features
        print(f'{sys._getframe().f_code.co_name} finish')
        
    def get_multiway_way_important_data(self):
        df = pd.DataFrame({'Variable': self.features, 'Node purity increase': self.impurity_decrease, 'MSE_increase': self.mse_increase, 'P_value': self.p_values, 'Top': self.top_features})
        df.to_csv(f'{self.save_path}/multiway_way_data.csv', index=False)
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_rf_tree_depth_analyse(self):
        try:
            n_trees = len(self.model.estimators_)
            min_depths = {feature: [] for feature in self.X_train.columns}
            # 计算每个特征在每棵树中的最小深度
            for tree in self.model.estimators_:
                for feature_idx, feature in enumerate(self.X_train.columns):
                    # 获取每个特征的最小深度
                    tree_depth = tree.tree_.max_depth
                    feature_depth = min([d for d, f in zip(tree.tree_.feature, tree.tree_.threshold) if f == feature_idx] or [tree_depth])
                    min_depths[feature].append(feature_depth)
            # 计算每个特征最小深度的平均值
            mean_min_depths = {feature: np.mean(depths) for feature, depths in min_depths.items()}
            # 创建结果DataFrame
            results = pd.DataFrame({
                'Feature': list(min_depths.keys()),
                'Mean_MinDepth': list(mean_min_depths.values())
            })
            for i in range(n_trees):
                results[f'Tree{i+1}_MinDepth'] = [min_depths[feature][i] for feature in self.X_train.columns]
            df = pd.DataFrame(results)
            df.to_csv(f'{self.save_path}/distribution_of_depth.csv', index=False)

        except Exception as e:
            print(f'这个模型{self.method}获取不了关于树的深度分布的~~~:', e)
        print(f'{sys._getframe().f_code.co_name} finish')



