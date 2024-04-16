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
    def __init__(self, X_train, y_train, X_test, y_test, kf, model_save_file, target, method):
        super().__init__(X_train, y_train, X_test, y_test, kf, model_save_file, target, method)
        self.model = RandomForestRegressor(n_estimators=1000, criterion='mse',
                    max_depth=10, min_samples_split=2,
                    min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                    max_features='auto',  max_leaf_nodes=None,
                    bootstrap=True, oob_score=False,
                    n_jobs=1, random_state=None,
                    verbose=0, warm_start=False)
        self.save_path = f'{model_save_file}/{method}'
        os.makedirs(self.save_path, exist_ok=True)

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

    def get_important_analyse(self):
        feat_importance=pd.DataFrame()
        feat_importance["Importance"]=self.model.feature_importances_
        feat_importance.set_index(self.X_train.columns, inplace=True)

        feat_importance = feat_importance.sort_values(by='Importance',ascending=True)
        pal=sns.color_palette("plasma_r", 70).as_hex()[2:]

        fig=go.Figure()
        for i in range(len(feat_importance.index)):
            fig.add_shape(dict(type="line", y0=i, y1=i, x0=0, x1=feat_importance['Importance'][i], 
                            line_color=pal[::-1][i],opacity=0.7,line_width=4))
        fig.add_trace(go.Scatter(x=feat_importance['Importance'], y=feat_importance.index, mode='markers', 
                                marker_color=pal[::-1], marker_size=8,
                                hovertemplate='%{y} Importance = %{x:.5f}<extra></extra>'))
        fig.update_layout(template=temp,title='Overall Feature Importance', 
                        xaxis=dict(title='Average Importance',zeroline=False),
                        yaxis_showgrid=False, margin=dict(l=120,t=80),
                        height=700, width=800)
        # 保存为HTML文件
        fig.write_html(f"{self.save_path}/feature_importance.html")

        # 保存为图像文件
        # fig.write_image(f"{self.save_path}/feature_importance.png")
        print('save ok-----------')