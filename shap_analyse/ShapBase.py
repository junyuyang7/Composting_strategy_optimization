import shap
import pandas as pd
from models import ModelBase, LGBTraining, CatTraining, RFTraining, GSRTraining, MLPTraining, XGBTraining, SVRTraining, LRTraining, RidgeTraining
import joblib
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import seaborn as sns
matplotlib.use("Agg")

# Material_Main,Turning times,V2_Ventilation Interval (min),M1_is Enclosed, Composting Method,Additive_4,Additive_2,Additive_1,Material_3,Material_2,Additive_3,Period (d),V1_Ventilation Type,V4_Ventilation Day,V3_Ventilation Duration (min),Compost volume (m3),Application Rate (%DW),V5_Ventilation rate (L/min/kg iniDW),Initial C/N (%),Initial Moisture Content (%),Initial TN (%),Initial pH
map_col = {
    "Application Rate (%DW)": "AA",
    "Material_Main": "WT",
    "Turning times": "TF",
    "V2_Ventilation Interval (min)": "IAF",
    " Composting Method": "CS",
    "Additive_4": "A4",
    "Additive_2": "A2",
    "Additive_1": "AT",
    "Material_3": "M3",
    "Material_2": "M2",
    "Additive_3": "A3",
    "Period (d)": "CD",
    "V1_Ventilation Type": "AM",
    "V4_Ventilation Day": "AD",
    "V3_Ventilation Duration (min)": "V3",
    "Compost volume (m3)": "PS",
    "V5_Ventilation rate (L/min/kg iniDW)": "AR",
    "Initial GI (%)": "IGI",
    "waste type": "WT",
    "bulking agent type": "BA",
    "Initial Moisture Content (%)": "IMC",
    "Initial TC (%)": "ITC",
    "Initial TN (%)": "ITN",
    "Initial C/N (%)": "ICN",
    "Initial pH": "IPH",
    "initial bulk density": "IBD",
    "composting systems": "CS",
    "aeration method": "AM",
    "turning frequency": "TF",
    "aeration rate": "AR",
    "intermittent aeration frequency": "IAF",
    "aeration duration": "AD",
    "composting duration": "CD",
    "covered": "CO",
    "pile size": "PS",
    "additive type": "AT",
    "additive amount": "AA",
    "germination index": "GI",
    "M1_is Enclosed": "CO",
}
remove_lst = ['A2', 'A3', 'A4', 'V3', 'M2', 'M3']

class ShapAnalyse:
    def __init__(self, X_train, y, seed,
                 target: str, 
                 model_name: str, 
                 model_path: str,
                 save_path: str):
        self.target = target
        self.model_name = model_name
        self.model_path = model_path
        self.models_dict = {'rf': RFTraining,
                'xgb': XGBTraining,
                'lgb': LGBTraining,
                'cat': CatTraining,
                'lr': LRTraining,
                'ridgelr': RidgeTraining,
                'mlp': MLPTraining,
                'svr': SVRTraining,
                'gsr': GSRTraining}
        self.X_train = X_train
        self.y = y
        self.cols = X_train.columns
        self.input_path = save_path
        self.save_path = f'{save_path}\\shap_plot\\{target}_{model_name}\\'
        self.seed = seed
        os.makedirs(self.save_path, exist_ok=True)
    
    def get_model(self):
        modelClass = self.models_dict[self.model_name](is_bayesian=False)
        self.model = modelClass.model
        self.model = joblib.load(self.model_path)
        if self.model_name in ['rf', 'xgb', 'lgb', 'cat']:
            try:
                self.explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.explainer.shap_values(self.X_train)
            except Exception as e:
                print(e)
                self.explainer = shap.TreeExplainer(self.model, data=self.X_train)
                self.shap_values = self.explainer.shap_values(self.X_train)
        else:
            self.explainer = shap.KernelExplainer(self.model.predict, self.X_train)
            self.shap_values = self.explainer.shap_values(self.X_train)
        shap_df = pd.DataFrame(self.shap_values, columns=self.X_train.columns)
        shap_df.to_csv(f'{self.save_path}{self.target}_shap_value.csv', index=False)
        self.s_values = self.explainer(self.X_train)
        self.expected_value = self.explainer.expected_value
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_featrue(self):
        cols = [map_col[col] for col in self.cols]
        
        plt.figure(figsize=(15, 5))
        feat_importance=pd.DataFrame()
        feat_importance["Importance"]=self.model.feature_importances_
        feat_importance.set_index(self.cols, inplace=True)
        feat_importance = feat_importance.sort_values(by='Importance', ascending=False)

        plt.bar(range(len(cols)), feat_importance['Importance'])
        plt.xticks(range(len(cols)), cols, rotation=90, fontsize=14)
        plt.title('Feature importance', fontsize=14)
        plt.savefig(f'{self.save_path}{self.target}_feature_importance.png')
        print(f'{sys._getframe().f_code.co_name} finish')
        

    def get_force_plot(self):
        fp = shap.force_plot(self.explainer.expected_value, self.shap_values, self.X_train)
        shap.save_html(f'{self.save_path}{self.target}_force_plot.html', fp)
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_feature_more(self):
        plt.figure(figsize=(10, 30))
        X_train = self.X_train.rename(columns=map_col)
        cols = X_train.columns.tolist()
        indices = [cols.index(item) for item in remove_lst if item in cols]
        shap_values = np.delete(self.shap_values, indices, axis=1)
        X_train = X_train.drop(columns=remove_lst)
        
        shap.summary_plot(shap_values, X_train, show=False)
        plt.savefig(f'{self.save_path}{self.target}_feature_importance_v1.png')
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_feature_bar(self):
        plt.figure(figsize=(10, 30))
        shap.summary_plot(self.shap_values, self.X_train, plot_type='bar', show=False)
        plt.savefig(f'{self.save_path}{self.target}_feature_importance_v2.png')
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_subplot(self):
        input_cols = self.cols.tolist()
        fig, axes = plt.subplots(len(input_cols)//3+1, 3, figsize=(30,90))
        for i, col in enumerate(input_cols):
            shap.dependence_plot(col, self.shap_values, self.X_train, interaction_index=None, show=False, ax=axes[i//3,i%3])
        plt.tight_layout()
        plt.savefig(f'{self.save_path}{self.target}shap_dependence_plots.png', bbox_inches='tight')
        plt.close()
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_interaction_plot(self):
        plt.figure(figsize=(24,12))
        interaction_values = self.explainer.shap_interaction_values(self.X_train)
        shap.summary_plot(interaction_values, self.X_train, max_display=len(self.cols)//4, show=False)
        plt.show()
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_dependence_plot(self, col1, col2:str):
        plt.figure(figsize=(10, 30))
        input_cols = self.cols.tolist()
        assert col1 in input_cols and col2 in input_cols, f"get_dependence_plot error: {col1} or {col2} not in input_cols"
        shap.dependence_plot(col1, self.shap_values, self.X_train, interaction_index=col2, show=False)
        col1 = col1.replace("/","_")
        col2 = col2.replace("/","_")
        plt.savefig(f'{self.save_path}{self.target}_{col1}_{col2}_dependence_plot.png')
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_decision_plot(self,):
        shap.decision_plot(self.expected_value, self.shap_values, self.X_train, show=False, ignore_warnings=True)
        plt.savefig(f'{self.save_path}{self.target}_decision_plot.png')
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_r2_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        input_path = f'{self.input_path}/model_{self.target}/{self.model_name}/{self.model_name}_pred_te.csv'
        df = pd.read_csv(input_path)
        y = df['true'].tolist()
        y_pred = df['pred'].tolist()
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        sns.scatterplot(x=y, y=y_pred, ax=ax)
        ax.plot([0, 40], [0, 40], 'k--')
        ax.set_title(f"{self.model_name}\n$R^2 = {r2:.2f}$\nRMSE = {rmse:.2f}")
        ax.set_xlabel(f"Ture value {self.target}")
        ax.set_ylabel(f"Predicted value {self.target}")
        ax.plot([min(y), max(y)], [min(y), max(y)], 'k--')

        plt.tight_layout()
        output_file = f'{self.input_path}/r2_plot/{self.target}'
        os.makedirs(output_file, exist_ok=True)
        plt.savefig(f'{output_file}/{self.model_name}_r2_plot.png')
        print(f'{sys._getframe().f_code.co_name} finish')

    # 保存特征交互矩阵
    def get_fea_inter(self):
        try:
            interaction_values = self.explainer.shap_interaction_values(self.X_train)
            mean_interaction_values = np.abs(interaction_values).mean(axis=0)
            feature_names = self.cols.tolist()
            interaction_df = pd.DataFrame(mean_interaction_values, columns=feature_names, index=feature_names)
            interaction_df.to_csv(f'{self.save_path}{self.target}_feaInter.csv')
        except Exception as e:
            print(e)
        print(f'{sys._getframe().f_code.co_name} finish')

    # 特征和目标值的相关性
    def get_feature_target(self):
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        feature_names = self.cols.tolist()
        shap_df = pd.DataFrame({'feas': feature_names, 'Correlation with Target': feature_importance})
        # 计算每个特征的SHAP值与目标值之间的相关性
        # correlation_scores = shap_df.apply(lambda col: col.corr(pd.Series(self.y)), axis=0)
        # correlation_df = pd.DataFrame({'feas': feature_names, 'Correlation with Target': correlation_scores})
        shap_df.to_csv(f'{self.save_path}{self.target}_feaTarget.csv')
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_heatmap_value(self):
        interaction_values = self.explainer.shap_interaction_values(self.X_train)
        # 计算平均交互值
        mean_interaction_values = np.mean(np.abs(interaction_values), axis=0)
        feature_names = self.cols.tolist()
        interaction_df = pd.DataFrame(mean_interaction_values, columns=feature_names, index=feature_names)
        interaction_df.to_csv(f'{self.save_path}{self.target}_feaHeatmap.csv')
        print(f'{sys._getframe().f_code.co_name} finish')


