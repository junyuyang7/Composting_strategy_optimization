import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
import json
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# 设置日志
from utils import set_logger, set_seed, set_filename, get_data, get_r2_compare
import logging
from models import ModelBase, RFTraining, LGBTraining, XGBTraining, CatTraining, LRTraining, \
                RidgeTraining, MLPTraining, SVRTraining, GSRTraining
from shap_analyse.ShapBase import ShapAnalyse


# seed=72
warnings.filterwarnings('ignore')  
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文  

output_file = 'output/data_selected_0612/__05' # 输出文件夹
input_file = 'data/data_selected_0612/__05' # 输入数据文件夹
isBayesian = True

os.makedirs(output_file, exist_ok=True)
log_path = output_file + f"/train.log" # 日志路径
# 设置日志和种子数
os.makedirs(os.path.dirname(log_path), exist_ok=True)
set_logger(log_path)
# set_seed(seed=seed)
sns.set(font='Microsoft YaHei')

# ================================参数设置===================================
# targets = ['TN loss (%)', 'NH3-N (g)', 'N2O-N (g)', 'NH3-N loss (%)', 'N2O-N loss (%)', 'TC loss (%)', 'CH4-C (g)', 'CO2-C (g)', 'CH4-C loss (%)', 'CO2-C loss (%)']
if int(input_file[-1]) % 2 == 1:
    targets = ['TN loss (%)','NH3-N loss (%)', 'N2O-N loss (%)', 'TC loss (%)', 
            'CH4-C loss (%)', 'CO2-C loss (%)']
else:
    targets =  [
        "Final Moisture Content (%)", 
        "Final pH", 
        "Final TN (%)", 
        "Final TC (%)", 
        "Final C_N (%)", 
        "Final EC (ms_cm)", 
        "Final GI (%)", 
        "Final NH3-N (g_kg)", 
        "Final NO2-N (g_kg)"
    ]
# targets = ['TC loss (%)', 'CH4-C (g)', 'CO2-C (g)', 'CH4-C loss (%)', 'CO2-C loss (%)']

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
models = {'rf': RFTraining,
          'xgb': XGBTraining,
          'lgb': LGBTraining,
          'cat': CatTraining,
          'lr': LRTraining,
          'ridgelr': RidgeTraining,
          'mlp': MLPTraining,
          'svr': SVRTraining,
          'gsr': GSRTraining}
use_models = ['rf', 'xgb', 'lgb', 'cat', 'lr', 'ridgelr',  'svr', 'gsr']
# use_models = [ 'mlp', 'svr', 'gsr']
use_models = ['mlp']

# ================================正式训练===================================
# 训练模型并挑选表现最好的模型进行shap值分析
for target in targets:
    if target in ['TN loss (%)', 'N2O-N (g)', 'NH3-N (g)', 'NH3-N loss (%)']:
        seed=2345
    else:
        seed=72
    set_seed(seed=seed)
    # 获取字典保存各个模型的最终结果
    logging.info('{} Prediction Training--------------------------------------------------------------------------------------'.format(target))
    print('{} Prediction Training--------------------------------------------------------------------------------------'.format(target))
    result_mse = {}
    result_r2 = {}
    result_mae = {}
    data_path, model_performance_path, mse_json, mae_json, r2_json, model_save_file = set_filename(target, output_file, input_file)
    X_train, X_test, y_train, y_test, input_cols = get_data(data_path, seed)
    for model_name in use_models:
        # 贝叶斯优化类
        
        model: ModelBase = models[model_name](X_train, y_train, X_test, y_test, \
                                        kf, model_save_file, target, method=model_name)
        if isBayesian:
            model.isBayesian()
        else:
            model.train()
        mse, mae, r2, pred = model.test()
        model.save_result()
        if model_name in ['rf', 'xgb', 'lgb', 'cat']:
            model.get_important_analyse()

        result_mse[f'{model_name}'] = metrics.mean_squared_error(y_test,pred)
        result_mae[f'{model_name}'] = metrics.mean_absolute_error(y_test,pred)
        result_r2[f'{model_name}'] = metrics.r2_score(y_test,pred)
    
    # 将字典保存为 JSON 文件
    with open(mse_json, 'w') as json_file:
        json.dump(result_mse, json_file)
    with open(mae_json, 'w') as json_file:
        json.dump(result_mae, json_file)
    with open(r2_json, 'w') as json_file:
        json.dump(result_r2, json_file)

    # 进行shap值分析
    best_model_name = max(result_r2, key=result_r2.get)
    print('the best model is {}'.format(best_model_name))
    logging.info('the best model is {}'.format(best_model_name))
    get_r2_compare(target=target, output_file=output_file, input_file=input_file)
    shapShower = ShapAnalyse(X_train=X_train, 
                             target=target,
                             model_name=best_model_name,
                             model_path=f'{model_save_file}/{best_model_name}/{best_model_name}_model.pkl',
                             save_path=f'{output_file}')
    shapShower.get_model()
    if best_model_name in ['rf', 'xgb', 'lgb', 'cat']:
        shapShower.get_featrue()
    shapShower.get_force_plot()
    shapShower.get_feature_more()
    shapShower.get_feature_bar()
    shapShower.get_subplot()
    shapShower.get_dependence_plot(input_cols[0], input_cols[1])
    shapShower.get_decision_plot()
    shapShower.get_r2_plot()
    # shapShower.get_interaction_plot()

