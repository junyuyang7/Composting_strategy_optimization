import shap
import pandas as pd
from models import ModelBase, LGBTraining, CatTraining, RFTraining, GSRTraining, MLPTraining, XGBTraining, SVRTraining, LRTraining, RidgeTraining
import joblib
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

class ShapAnalyse:
    def __init__(self, X_train, 
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
        self.cols = X_train.columns
        self.save_path = f'{save_path}/shap_plot/{target}_{model_name}/'
        os.makedirs(self.save_path, exist_ok=True)
    
    def get_model(self):
        modelClass = self.models_dict[self.model_name]()
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
        self.expected_value = self.explainer.expected_value
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_featrue(self):
        plt.figure(figsize=(15, 5))
        feat_importance=pd.DataFrame()
        feat_importance["Importance"]=self.model.feature_importances_
        feat_importance.set_index(self.cols, inplace=True)
        feat_importance = feat_importance.sort_values(by='Importance', ascending=False)

        plt.bar(range(len(self.cols)), feat_importance['Importance'])
        plt.xticks(range(len(self.cols)), feat_importance.index, rotation=90, fontsize=14)
        plt.title('Feature importance', fontsize=14)
        plt.savefig(f'{self.save_path}{self.target}_feature_importance.png')
        print(f'{sys._getframe().f_code.co_name} finish')
        

    def get_force_plot(self):
        fp = shap.force_plot(self.explainer.expected_value, self.shap_values, self.X_train)
        shap.save_html(f'{self.save_path}{self.target}_force_plot.html', fp)
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_feature_more(self):
        plt.figure(figsize=(10, 30))
        shap.summary_plot(self.shap_values, self.X_train, show=False)
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

    def get_dependence_plot(self, col1, col2):
        plt.figure(figsize=(10, 30))
        input_cols = self.cols.tolist()
        assert col1 in input_cols and col2 in input_cols, f"get_dependence_plot error: {col1} or {col2} not in input_cols"
        shap.dependence_plot(col1, self.shap_values, self.X_train, interaction_index=col2, show=False)
        plt.savefig(f'{self.save_path}{self.target}_{col1}_{col2}_dependence_plot.png')
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_decision_plot(self,):
        shap.decision_plot(self.expected_value, self.shap_values, self.X_train, show=False, ignore_warnings=True)
        plt.savefig(f'{self.save_path}{self.target}_decision_plot.png')
        print(f'{sys._getframe().f_code.co_name} finish')
    