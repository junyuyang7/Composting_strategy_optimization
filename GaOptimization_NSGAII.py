from GA_method.GAoptimation import GAOptimation
from GA_method.GAoptimation01 import NSGA_II
# from GA_method.GAoptimation01_mp import NSGA_II
from utils import get_best_model_name, get_inner_min_max, set_seed
import os
import json
import warnings
warnings.filterwarnings('ignore') 
set_seed(100)
output_file = r'output/Ga/raw_useall_True_1008' # 输出文件夹
input_file = r'data/Ga/0630_16' # 输入数据文件夹
model_file = r'output/raw_data_selected_0625_useall_False'

targets = ["Final GI (%)", 'NH3-N loss (%)', 'N2O-N loss (%)', 'CH4-C loss (%)', 'CO2-C loss (%)']
# targets = ['NH3-N loss (%)', 'N2O-N loss (%)', 'CH4-C loss (%)', 'CO2-C loss (%)']

# 设置参数
population_size = 50 # 40  50  80 100
n_generations = 20  # 20  10  40 30
cxpb = 0.2  # 交叉概率 0.5
mutpb = 0.03  # 变异概率 0.05
num_runs = 600  # 200
# ratio_map = {"Final GI (%)": 0.8}
int_feas = ['Material_Main', 'Material_2', 'Material_3', 'Additive Species', 'Additive_1','Additive_2', 
                    'Additive_3',"Additive_4",'Method', 'M1_is Enclosed', 'M2_is Turning', 'M3_isForce aeration',
                    'M4_isVessel', 'M5_isReactor', 'V1_Ventilation Type', 'V6_Extra', 
                    'Aeration method', 'Composting Method']
# 获取列名
# with open(f'output/select_cols.json', 'r') as json_file:
#     target_select_col = json.load(json_file)
target_select_col = {}

# 需要固定的组合
fix_fea = ['Material_Main']
# fix_nums_lst = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
x = int(input('输入Material_Main 的类别'))

fix_nums_lst = [[x]]
# 更新四个气体
for fix_nums in fix_nums_lst:
    model_names = []
    data_paths = []
    model_paths = []
    for i, target in enumerate(targets):
        try:
            model_file__16 = os.path.join(model_file,"__16")
            json_path = f'{model_file__16}/model_{target}/result_r2_{target}.json'
            data_path = f'{input_file}/data_for_{target}.csv'
            model_name = get_best_model_name(json_path)
            model_file_new = model_file__16.replace("False","True")
        except:
            model_file__18 = os.path.join(model_file,"__18")
            json_path = f'{model_file__18}/model_{target}/result_r2_{target}.json'
            data_path = f'{input_file}/data_for_{target}.csv'
            model_name = get_best_model_name(json_path)
            model_file_new = model_file__18.replace("False","True")
        
        print(f'{target} best_model is {model_name}')
        model_path = f'{model_file_new}/model_{target}/{model_name}/{model_name}_model.pkl'
        model_names.append(model_name)
        model_paths.append(model_path)
        data_paths.append(data_path)

    print(f'--------------Optimizate {targets}------------------')
    NSGa_client = NSGA_II(model_names, model_paths, data_paths, output_file, targets, target_select_col, fix_nums)
    NSGa_client.get_model()
    raw_dict = NSGa_client.get_min_max()
    with open(f'{output_file}/GaReasult/min_max_{targets}_final.json', 'w') as f:
        json.dump(raw_dict, f)

    NSGa_client.optimization(targets, 
                            population_size=population_size, 
                            n_generations=n_generations, 
                            cxpb=cxpb, 
                            mutpb=mutpb, 
                            num_runs=num_runs,
                            new_dict=raw_dict)
