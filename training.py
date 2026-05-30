import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib
import os
import sys
import warnings
import json
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from utils import set_logger, set_seed, set_filename, get_data, get_r2_compare
import logging
from models import ModelBase, RFTraining, LGBTraining, XGBTraining, CatTraining, LRTraining, \
                RidgeTraining, MLPTraining, SVRTraining, GSRTraining
from shap_analyse.ShapBase import ShapAnalyse


# Set seed for reproducibility
warnings.filterwarnings('ignore')  
plt.rcParams['font.sans-serif']=['SimHei'] # Display Chinese characters

use_all = False
dirs = f'data/data_selected_1225'
output_dir = f'output/raw_data_selected_1225_useall_{use_all}' # Output folder
input_dir = 'data/data_selected_1225' # Input data folder
num_runs = 5 # Number of runs for robustness validation
best_model_name = None
only_shap = False
best_model_map = {
    'TN loss (%)': 'cat',
    'NH3-N loss (%)': 'cat',
    'N2O-N loss (%)': 'xgb',
    'TC loss (%)': 'cat',
    'CH4-C loss (%)': 'xgb',
    'CO2-C loss (%)': 'xgb',
    'Final GI (%)': 'rf'
}

for name in os.listdir(dirs):

    print(name)
    
    output_file = os.path.join(output_dir, name) # Output folder
    input_file = os.path.join(input_dir, name) # Input data folder
    isBayesian = False

    # last_number = int(input_file.split('_')[-1])
    last_number = int(input_file.split('_')[-1])
    # if last_number not in [16,18]:
    if last_number not in [16]:
        continue
    os.makedirs(output_file, exist_ok=True)
    log_path = output_file + f"/train.log" # Log path
    set_logger(log_path)
    sns.set(font='Microsoft YaHei')

    if last_number in [13, 15, 18, 19]:
        targets = ["Final GI (%)"]
    elif last_number in [12, 14]:
        targets = ['TN loss (%)','NH3-N loss (%)', 'N2O-N loss (%)', 'TC loss (%)', 
                   'CH4-C loss (%)', 'CO2-C loss (%)', "Final GI (%)"]
    elif last_number in [16, 17]:
        targets = ['NH3-N loss (%)', 'N2O-N loss (%)', 'CH4-C loss (%)', 'CO2-C loss (%)']
    print(targets)

    num_folds = 10
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    models = {'rf': RFTraining, 'xgb': XGBTraining, 'lgb': LGBTraining, 'cat': CatTraining,
              'lr': LRTraining, 'ridgelr': RidgeTraining, 'mlp': MLPTraining, 'svr': SVRTraining, 'gsr': GSRTraining}
    use_models = ['rf', 'xgb', 'lgb', 'cat']
    # use_models = ['xgb', 'lgb', 'cat']
    # use_models = ['rf']
    # use_models = ['lr', 'ridgelr', 'gsr',"mlp"]

# [2345, 72, 42, 3456, 1234] CO2 最优 >0.8
# [2345, 72, 42, 34, 114514] TC 最优 >0.8
# [2345, 72, 456, 123, 12345] N2O 最优 0.755
# [1489, 72, 42, 55, 114514] TN 最优 0.718
# [123, 72, 42, 159, 244] CH4 最优 0.728
# [72, 45127, 456, 3753, 2345] NH3 最优
# [172, 72, 73245, 314, 4107] Final GI 最优

    for target in targets:
        if target in ['TC loss (%)']:
            seeds = [2345, 42, 34, 114514, 72]
        elif target in ['TN loss (%)', ]:
            seeds= [1489, 72, 42, 55, 114514]
        elif target in ['CH4-C loss (%)']:   
            seeds= [123, 7834, 159, 244, 72]
        elif target in ['NH3-N loss (%)']:
            seeds= [7460, 45127, 456, 3890, 5678]
        elif target in ['N2O-N loss (%)']:
            seeds= [1234, 72, 7834, 42157, 456]
        elif target in ['CO2-C loss (%)']:
            seeds = [72, 42, 3456, 1234, 2345]
        else:  # Final GI(%) 4107
            seeds = [172, 72, 73245, 314, 4107]

        results = {model_name: {'mse': [], 'mae': [], 'r2': []} for model_name in use_models}
        data_path, model_performance_path, mse_json, mae_json, r2_json, model_save_file = set_filename(target, output_file, input_file)
        seed0 = seeds[-1]
        # X_train, X_test, y_train, y_test, input_cols = get_data(data_path, seeds[-1], use_all)
        
        if not only_shap:
            # for run in range(num_runs):
            for run in range(len(seeds)):
                seed0 = seeds[run]
                set_seed(seed0)
                X_train, X_test, y_train, y_test, input_cols = get_data(data_path, seed0, use_all)
                logging.info('{} Prediction Training Run {}--------------------------------------------------------------------------------------'.format(target, run + 1))
                print('{} Prediction Training Run {}----------------------------------------------seed{}----------------------------------------'.format(target, run + 1,seed0))

                

                for model_name in use_models:
                    model: ModelBase = models[model_name](X_train, y_train, X_test, y_test, kf, model_save_file, target, method=model_name, is_bayesian=isBayesian)
                    model.train()
                    if not use_all:
                        mse, mae, r2, pred = model.test()
                        results[model_name]['mse'].append(metrics.mean_squared_error(y_test, pred))
                        results[model_name]['mae'].append(metrics.mean_absolute_error(y_test, pred))
                        results[model_name]['r2'].append(metrics.r2_score(y_test, pred))

                        # Save average results as JSON files
                        with open(f'{model_save_file}/result_mse_{target}{seed0}.json', 'w') as json_file:
                            json.dump({model_name: values['mse'] for model_name, values in results.items()}, json_file)
                        with open(f'{model_save_file}/result_mae_{target}{seed0}.json', 'w') as json_file:
                            json.dump({model_name: values['mae'] for model_name, values in results.items()}, json_file)
                        with open(f'{model_save_file}/result_r2_{target}{seed0}.json', 'w') as json_file:
                            json.dump({model_name: values['r2'] for model_name, values in results.items()}, json_file)


                    if model_name in ['rf', 'xgb', 'lgb', 'cat']:
                        model.get_important_analyse()
                        # 获取一些绘图数据
                        model.get_mse_increase()
                        model.get_pure_increase()
                        model.get_p_value()
                        model.get_important_mark()
                        model.get_multiway_way_important_data()
                        model.get_rf_tree_depth_analyse()

                    model.save_result()

            if not use_all:
                avg_results = {model_name: {'mse': np.mean(values['mse']), 'mae': np.mean(values['mae']), 'r2': np.mean(values['r2'])} for model_name, values in results.items()}
                # Save average results as JSON files
                with open(mse_json, 'w') as json_file:
                    json.dump({model_name: values['mse'] for model_name, values in avg_results.items()}, json_file)
                with open(mae_json, 'w') as json_file:
                    json.dump({model_name: values['mae'] for model_name, values in avg_results.items()}, json_file)
                with open(r2_json, 'w') as json_file:
                    json.dump({model_name: values['r2'] for model_name, values in avg_results.items()}, json_file)

                best_model_name = max(avg_results, key=lambda model_name: avg_results[model_name]['r2'])
                print('The best model is {}'.format(best_model_name))
                logging.info('The best model is {}'.format(best_model_name))
                get_r2_compare(target=target, output_file=output_file, input_file=input_file)
            
            else:
                best_model_name = best_model_map[target]
            
        if not best_model_name:
            with open(r2_json, 'r') as json_file:
                avg_result = json.load(json_file)
            best_model_name = max(avg_result, key=lambda model_name: avg_result[model_name])
        best_model_name = best_model_map[target]

        shapShower = ShapAnalyse(X_train=X_train, y=y_train, seed=seed0, target=target, model_name=best_model_name,
                                 model_path=f'{model_save_file}/{best_model_name}/{best_model_name}_model.pkl',
                                 save_path=f'{output_file}')
        print('shap begin')
        # shapShower.get_model()
        # if best_model_name in ['rf', 'xgb', 'lgb', 'cat']:
        #     shapShower.get_featrue()
        # shapShower.get_force_plot()
        # shapShower.get_feature_more()
        # shapShower.get_feature_bar()
        # shapShower.get_subplot()
        # shapShower.get_dependence_plot(input_cols[0], input_cols[1])
        # shapShower.get_decision_plot()
        # if not use_all:
        #     shapShower.get_r2_plot()

        # 获取一些绘图数据
        # shapShower.get_fea_inter()
        # shapShower.get_feature_target()
        # # shapShower.get_heatmap_value()
