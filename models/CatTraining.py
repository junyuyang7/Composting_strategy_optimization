from catboost import CatBoostRegressor
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

class CatTraining(ModelBase):
    def __init__(self, X_train=None, 
                 y_train=None, 
                 X_test=None, 
                 y_test=None, 
                 kf=None, 
                 model_save_file=None, 
                 target=None, 
                 method=None):
        super().__init__(X_train, y_train, X_test, y_test, kf, model_save_file, target, method)
        params = {
            'learning_rate': 0.05,
            'loss_function': "RMSE",
            'eval_metric': "RMSE", # CrossEntropy
            'depth': 8,
            'min_data_in_leaf': 20,
            'random_seed': 42,
            'logging_level': 'Silent',
            'use_best_model': True,
            'one_hot_max_size': 5,   #类别数量多于此数将使用ordered target statistics编码方法,默认值为2。
            'boosting_type':"Ordered", #Ordered 或者Plain,数据量较少时建议使用Ordered,训练更慢但能够缓解梯度估计偏差。
            'max_ctr_complexity': 2, #特征组合的最大特征数量，设置为1取消特征组合，设置为2只做两个特征的组合,默认为4。
            'nan_mode': 'Min' 
        }
        iterations = 400
        early_stopping_rounds = 200
        self.model = CatBoostRegressor()

        self.save_path = f'{model_save_file}/{method}'
        os.makedirs(self.save_path, exist_ok=True)

    # def train(self):
    #     X_train, y_train, kf = self.X_train, self.y_train, self.kf

    #     preds = np.zeros(X_train.shape[0])

    #     logging.info('{} Training Begin--------------------------------------------------------------------------------------'.format(self.method))
    #     for i, (train_index, valid_index) in enumerate(kf.split(X_train)):
    #         x_tr, x_va = X_train.iloc[train_index], X_train.iloc[valid_index]
    #         y_tr, y_va = y_train.iloc[train_index], y_train.iloc[valid_index]

    #         self.model.fit(x_tr, y_tr, eval_set=(x_va, y_va), verbose=0, early_stopping_rounds=1000)
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


    # def test(self):
    #     X_test, y_test, model_save_file = self.X_test, self.y_test, self.model_save_file
        
    #     pred_test = self.model.predict(X_test)
    #     self.pred_test = pred_test

    #     self.mse = metrics.mean_squared_error(y_test, pred_test)
    #     self.mae = metrics.mean_absolute_error(y_test, pred_test)
    #     self.r2 = metrics.r2_score(y_test, pred_test)
    #     logging.info('Test: The rmse of the {} is: {}'.format(self.method, self.mse))
    #     logging.info('Test: The mae of the {} is: {}'.format(self.method, self.mae))
    #     logging.info('Test: The R2 of the {} is: {}'.format(self.method, self.r2))

    #     x = list(range(0, len(y_test)))
    #     plt.figure(figsize=(40,10))
    #     plt.plot(x[:100], y_test[:100], 'g', label='ground-turth')
    #     plt.plot(x[:100], pred_test[:100], 'r', label=f'{self.method} predict')
    #     plt.legend(loc='best')
    #     plt.title({self.target})
    #     plt.xlabel('sample')
    #     plt.ylabel('Yield_lnRR')
    #     plt.savefig(f'{self.save_path}/plot_test.png')
    #     plt.close()

    #     return self.mse, self.mae, self.r2, self.pred_test
    
    # def save_result(self):
    #     joblib.dump(self.model, f'{self.save_path}/{self.method}_model.pkl') 
    #     df_tmp = pd.DataFrame({'pred': self.pred_test, 'true': self.y_test})
    #     df_tmp.to_csv(f'{self.save_path}/{self.method}_pred.csv', index=False)

    # def get_important_analyse(self):
    #     feat_importance=pd.DataFrame()
    #     feat_importance["Importance"]=self.model.feature_importances_
    #     feat_importance.set_index(self.X_train.columns, inplace=True)

    #     feat_importance = feat_importance.sort_values(by='Importance',ascending=True)
    #     pal=sns.color_palette("plasma_r", 70).as_hex()[2:]

    #     fig=go.Figure()
    #     for i in range(len(feat_importance.index)):
    #         fig.add_shape(dict(type="line", y0=i, y1=i, x0=0, x1=feat_importance['Importance'][i], 
    #                         line_color=pal[::-1][i],opacity=0.7,line_width=4))
    #     fig.add_trace(go.Scatter(x=feat_importance['Importance'], y=feat_importance.index, mode='markers', 
    #                             marker_color=pal[::-1], marker_size=8,
    #                             hovertemplate='%{y} Importance = %{x:.5f}<extra></extra>'))
    #     fig.update_layout(template=temp,title='Overall Feature Importance', 
    #                     xaxis=dict(title='Average Importance',zeroline=False),
    #                     yaxis_showgrid=False, margin=dict(l=120,t=80),
    #                     height=700, width=800)
    #     # 保存为HTML文件
    #     fig.write_html(f"{self.save_path}/feature_importance.html")

    #     # 保存为图像文件
    #     # fig.write_image(f"{self.save_path}/feature_importance.png")
    #     print(f'{self.__class__.__name__} save ok...')