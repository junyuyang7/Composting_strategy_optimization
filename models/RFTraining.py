from sklearn.ensemble import RandomForestRegressor
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
from models.ModelBase import ModelBase
import pandas as pd
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12), width=800))
colors=px.colors.qualitative.Plotly

class RFTraining(ModelBase):
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
        super().__init__(X_train, y_train, X_test, y_test, kf, model_save_file, target, method)
        if is_bayesian:
            self.model = RandomForestRegressor()
            self.isBayesian()
        else:
            # self.model = RandomForestRegressor(n_estimators=1000, criterion='friedman_mse',
            #                 max_depth=15, min_samples_split=2,
            #                 min_samples_leaf=1, min_weight_fraction_leaf=0.0,
            #                 max_leaf_nodes=None,
            #                 bootstrap=True, oob_score=False,
            #                 n_jobs=1, random_state=None,
            #                 verbose=0, warm_start=False)
            self.model = RandomForestRegressor()
        # self.save_path = f'{model_save_file}/{method}'
        os.makedirs(self.save_path, exist_ok=True)

    # def train(self):
    #     X_train, y_train, kf = self.X_train, self.y_train, self.kf

    #     preds = np.zeros(X_train.shape[0])

    #     logging.info('{} Training Begin--------------------------------------------------------------------------------------'.format(self.method))
    #     for i, (train_index, valid_index) in enumerate(kf.split(X_train)):
    #         x_tr, x_va = X_train.iloc[train_index], X_train.iloc[valid_index]
    #         y_tr, y_va = y_train.iloc[train_index], y_train.iloc[valid_index]

    #         self.model.fit(x_tr, y_tr)
    #         pred_valid = self.model.predict(x_va)
    #         preds[valid_index] = pred_valid

    #         logging.info('{}, {}'.format(x_tr.shape, y_tr.shape))
    #         logging.info('Train: The rmse of the {} is: {}'.format(self.method, metrics.mean_squared_error(y_va,pred_valid)))
    #         logging.info('Train: The mae of the {} is: {}'.format(self.method, metrics.mean_absolute_error(y_va,pred_valid)))
    #         logging.info('Train: The R2 of the {} is: {}'.format(self.method, metrics.r2_score(y_va,pred_valid)))
    #         x = list(range(0, len(y_va)))
    #         plt.figure(figsize=(40,10))
    #         plt.plot(x, y_va, 'g', label='ground-turth')
    #         plt.plot(x, pred_valid, 'r', label=f'{self.method} predict')
    #         plt.legend(loc='best')
    #         plt.title(f'{self.method}-yield')
    #         plt.xlabel('sample')
    #         plt.ylabel('Yield_lnRR')
    #         plt.savefig(f'{self.save_path}/plot_{i}.png')  # 保存图形为 plot_0.png, plot_1.png, ...
    #         plt.close()  # 关闭图形，以释放内存

    