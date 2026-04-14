import json
import joblib
import pandas as pd
from models import ModelBase, LGBTraining, CatTraining, RFTraining, GSRTraining, MLPTraining, XGBTraining, SVRTraining, LRTraining, RidgeTraining
from deap import base, creator, tools, algorithms
import random
import os
import sys
from utils import create_individual, evaluate, main, get_min_max_df, custom_mate,custom_mutate, get_inner_min_max
from functools import partial

# cat_feas = [
#             'Additive_4', 'Turning times', 'Material_2', 'V1_Ventilation Type', 'V2_Ventilation Interval (min)',
#             'Material_Main', 'Material_3', 'Composting Method', 'Additive_2', 'Aeration method', 'Additive_1',
#             'Composting Method', 'Additive_3', 'M1_is Enclosed'
#         ]
# common_fixed = [
#     'V3_Ventilation Duration (min)', 'Initial Moisture Content (%)', 'Period (d)', 'V1_Ventilation Type',
#     'V4_Ventilation Day', 'Additive_2', 'V2_Ventilation Interval (min)', 'Compost volume (m3)',
#     'Application Rate (%DW)', 'V5_Ventilation rate (L/min/kg iniDW)', 'Initial C/N (%)', 'Material_Main',
#     'M1_is Enclosed', 'Turning times', 'Additive_1', 'Material_2'
# ]

# # 定义特定的整型键列表
# specific_int_keys = [
#     'Additive_4', 'Turning times', 'Material_2', 'V1_Ventilation Type', 'V2_Ventilation Interval (min)',
#     'Material_Main', 'Material_3', 'Composting Method', 'Additive_2', 'Aeration method', 'Additive_1',
#     ' Composting Method', 'Additive_3', 'M1_is Enclosed'
# ]

int_feas = ['Material_Main', 'Material_2', 'Material_3', 'Additive Species', 'Additive_1','Additive_2', 
                    'Additive_3',"Additive_4",'Method', 'M1_is Enclosed', 'M2_is Turning', 'M3_isForce aeration',
                    'M4_isVessel', 'M5_isReactor', 'V1_Ventilation Type', 'V6_Extra', 
                    'Aeration method', 'Composting Method']

float_feas = ['Application Rate (%DW)', 'Initial Moisture Content (%)', 'Initial pH','Initial TN (%)',
                   'Initial TC (%)','Initial C/N (%)', 'Initial EC (ms/cm)', 'Initial GI (%)', 'Initial FW (kg)', 
                   'Initial DW (kg)', 'Initial NO2-N (g/kg)', 'Initial NH3-N (g/kg)', 'Compost volume (m3)', 
                   'Initial density (kg/L)', 'V3_Ventilation Duration (min)',
                 'V5_Ventilation rate (L/min/kg iniDW)']

int_feas_v1 = ['Period (d)', 'Turning times', 'V2_Ventilation Interval (min)', 'V4_Ventilation Day']

fix_fea = ['Material_Main']

class GAOptimation:
    def __init__(self, model_name, model_path, data_path, output_file,target, targets_cols):
        self.target = target
        # self.cat_feas = cat_feas
        self.model_name = model_name
        self.model_path = model_path
        self.output_file = output_file
        self.data_path = data_path
        self.df_raw = pd.read_csv(data_path)
        self.models_dict = {'rf': RFTraining,
            'xgb': XGBTraining,
            'lgb': LGBTraining,
            'cat': CatTraining,
            'lr': LRTraining,
            'ridgelr': RidgeTraining,
            'mlp': MLPTraining,
            'svr': SVRTraining,
            'gsr': GSRTraining}
        self.save_path = f'{output_file}/GaReasult/'
        self.targets_cols = targets_cols
        os.makedirs(self.save_path, exist_ok=True)

    def get_model(self):
        modelClass = self.models_dict[self.model_name](is_bayesian=False)
        self.model = modelClass.model
        self.model = joblib.load(self.model_path)
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_min_max(self):
        # 存储所有输入特征和标签的最小值和最大值的字典
        input_cols = list(self.df_raw.columns)
        min_max_values = get_min_max_df(self.df_raw, input_cols, self.output_file, int_feas=int_feas, int_feas_v1=int_feas_v1)

        min_max_json = json.dumps(min_max_values)
        with open(f'{self.output_file}/GaReasult/min_max_{self.target}.json', 'w') as f:
            f.write(min_max_json)

        self.raw_dict = min_max_values
        print(f'{sys._getframe().f_code.co_name} finish')
        return min_max_values

    # 从最大GI中去相应相同的特征值  键值对
    def create_individual(self, input_ranges, toolbox):
        individual = {}
        for attr_name, attr_info in input_ranges.items():
            if attr_name in int_feas:
                individual[attr_name] = toolbox.attr_int(attr_info)
            elif attr_name in int_feas_v1:
                individual[attr_name] = toolbox.attr_int_v1(attr_info)
            else:
                individual[attr_name] = toolbox.attr_float(attr_info)
        return creator.Individual(individual)
    
    # 保证整型的测试代码
    def evaluate(self, individual, input_ranges):
        # print(individual)
        individual_with_names = {attr_name: value for attr_name, value in zip(input_ranges.keys(), individual.values())}
        # 检查每个特征值是否超出范围，如果超出范围则返回一个非常大的适应度值
        for key, value in individual_with_names.items():
            if key in int_feas:
                if value not in input_ranges[key]:
                    return (-1e6,) if self.target == "Final GI (%)" else (1e6,)
            else:
                try:
                    if value < input_ranges[key][0] or value > input_ranges[key][1]:
                        return (-1e6,) if self.target == "Final GI (%)" else (1e6,)
                except:
                    pass


        # 使用模型进行预测
        individual_df = pd.DataFrame([individual_with_names], columns=self.targets_cols[self.target])
        # # 打印 individual_df 以进行调试
        # print("individual_df:")
        # print(individual_df)
        prediction = self.model.predict(individual_df)

        return (prediction,) if self.target == "Final GI (%)" else (-prediction,)
    
    def optimization(self, target, population_size=200, n_generations=100, cxpb=0.5, mutpb=0.01, num_runs=200, all_min_max=None):
        # 创建输入范围字典和输出范围字典
        input_ranges = {key: value for key, value in all_min_max.items() if key != target and key in list(self.df_raw.columns)}
        toolbox = base.Toolbox()
        
        if(self.target == "Final GI (%)"):
            # 最小化目标函数
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", dict, fitness=creator.FitnessMax)
        else:
            # 创建一个FitnessMax类，用于最大化目标函数
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", dict, fitness=creator.FitnessMin)
        
        # 定义生成整数属性的方法
        def attr_int(lst: list):
            return random.sample(lst, 1)[0]
        def attr_float(lst: list):
            return random.uniform(lst[0], lst[1])
        def attr_int_v1(lst: list):
            return random.randint(int(lst[0]), int(lst[1]+1))

        toolbox.register("attr_int", attr_int)
        toolbox.register("attr_float", attr_float)
        toolbox.register("attr_int_v1", attr_int_v1)

        # 注册个体生成方法
        toolbox.register("individual", self.create_individual, toolbox=toolbox, input_ranges=input_ranges)
        # 初始化种群
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate",  lambda ind: self.evaluate(ind, input_ranges=input_ranges))

        toolbox.register("mate", custom_mate, specific_keys=fix_fea)  # 交叉方式使用自定义交叉
        toolbox.register("mutate", custom_mutate, specific_keys=fix_fea)  # 变异方式使用自定义变异
        toolbox.register("select", tools.selTournament, tournsize=3)   
       
        # 开始训练 
        best_individuals = []
        for i in range(num_runs):
            pop, stats, hof = main(toolbox, self.target, population_size, n_generations, cxpb, mutpb)
            best_individual = tools.selBest(pop, 1)[0]
            print(f"Run {i+1}: Best individual with fitness {best_individual.fitness.values[0]}")
            
            best_individual_with_names = dict(best_individual)
            best_individuals.append((best_individual, best_individual_with_names))

        # 保存最优个体到CSV文件
        file_name = f"{self.save_path}individuals_{target}.csv"
        data = []
        columns = list(input_ranges.keys()) + [target]
        for best_individual, best_individual_with_names in best_individuals:
            row = [best_individual_with_names[attr] for attr in input_ranges.keys()]
            
            if(self.target != "Final GI (%)"):    
                row.append(-best_individual.fitness.values[0])  # 添加适应度值
            else:
                row.append(best_individual.fitness.values[0])    
            data.append(row)

        self.df = pd.DataFrame(data, columns=columns)
        self.df.to_csv(file_name, index=False)
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_optim_comb(self, target):
        min_combinations = []
        if(self.target != "Final GI (%)"):
            target_value = min(self.df[target])
        else:
            target_value = max(self.df[target])
        
        # GI要选大于百分之80以上的
        for index, row in self.df.iterrows():
            if row[target] == target_value:
                min_combinations.append(row)


        if(self.target != "Final GI (%)"):
            print(f"\nCombinations with minimum {target} ({target_value}):")
            min_combinations_df = pd.DataFrame(min_combinations)
            min_combinations_df.to_csv(f'{self.save_path}individuals_min_{target}.csv', index=False)
        else:
            print(f"\nCombinations with maximum {target} ({target_value}):")
            min_combinations_df = pd.DataFrame(min_combinations)
            min_combinations_df.to_csv(f'{self.save_path}individuals_max_{target}.csv', index=False)
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_input_range(self, target: str, ratio: int, is_inner: bool) -> dict:
        '''更新优化的输入特征最大最小值'''
        if target == 'Final GI (%)':
            acend = False
        else:
            acend = True
        self.df.sort_values(by=target, axis=0, ascending=acend, inplace=True)
        target_lst = self.df_raw[target].tolist()
        # threhold = (max(target_lst)-min(target_lst))*ratio + min(target_lst)
        if target == 'Final GI (%)':
            self.df = self.df[self.df[target] > 80]
        else:
            self.df.sort_values(by=[target], ascending=True, inplace=True)
            self.df = self.df[:len(self.df)*0.1]
        new_df = self.df.drop(target, axis=1)
        self.optim_dict = get_min_max_df(new_df, list(new_df.columns), output_file=self.output_file, int_feas=int_feas, int_feas_v1=int_feas_v1)
        if is_inner:
            raw_optim_dict = get_inner_min_max(self.raw_dict, self.optim_dict, int_feas)
        else:
            raw_optim_dict = self.optim_dict
        print(f'{sys._getframe().f_code.co_name} finish')
        return raw_optim_dict
        
