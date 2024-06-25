import json
import joblib
import pandas as pd
from models import ModelBase, LGBTraining, CatTraining, RFTraining, GSRTraining, MLPTraining, XGBTraining, SVRTraining, LRTraining, RidgeTraining
from deap import base, creator, tools, algorithms
import random
import os
import sys
from utils import create_individual, evaluate, main, get_min_max_df, custom_mate,custom_mutate
from functools import partial

cat_feas = [
            'Additive_4', 'Turning times', 'Material_2', 'V1_Ventilation Type', 'V2_Ventilation Interval (min)',
            'Material_Main', 'Material_3', 'Composting Method', 'Additive_2', 'Aeration method', 'Additive_1',
            'Composting Method', 'Additive_3', 'M1_is Enclosed'
        ]
common_fixed = [
    'V3_Ventilation Duration (min)', 'Initial Moisture Content (%)', 'Period (d)', 'V1_Ventilation Type',
    'V4_Ventilation Day', 'Additive_2', 'V2_Ventilation Interval (min)', 'Compost volume (m3)',
    'Application Rate (%DW)', 'V5_Ventilation rate (L/min/kg iniDW)', 'Initial C/N (%)', 'Material_Main',
    'M1_is Enclosed', 'Turning times', 'Additive_1', 'Material_2'
]



# 定义特定的整型键列表
specific_int_keys = [
    'Additive_4', 'Turning times', 'Material_2', 'V1_Ventilation Type', 'V2_Ventilation Interval (min)',
    'Material_Main', 'Material_3', 'Composting Method', 'Additive_2', 'Aeration method', 'Additive_1',
    ' Composting Method', 'Additive_3', 'M1_is Enclosed'
]

class GAOptimation:
    def __init__(self, model_name, model_path, data_path, output_file,target):
        self.target = target
        self.output_file = output_file
        self.cat_feas = cat_feas
        self.model_name = model_name
        self.model_path = model_path
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
        os.makedirs(self.save_path, exist_ok=True)

    def get_model(self):
        modelClass = self.models_dict[self.model_name]()
        self.model = modelClass.model
        self.model = joblib.load(self.model_path)
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_min_max(self):
        # 存储所有输入特征和标签的最小值和最大值的字典
        input_cols = list(self.df_raw.columns)
        # input_features = list(self.df_raw.columns[:-1])
        # label_column = self.df_raw.columns[-1]
        self.min_max_values = get_min_max_df(self.df_raw, input_cols,self.output_file)
        print(f'{sys._getframe().f_code.co_name} finish')
        
        
        
    # 保证整型的测试代码
    # def create_individual(self, input_ranges, toolbox):
    #     individual = []
    #     for attr_name, attr_info in input_ranges.items():
    #         if attr_name in self.cat_feas:
    #             individual.append(toolbox.attr_int(attr_info['Minimum'], attr_info['Maximum']))
    #         else:
    #             individual.append(toolbox.attr_float(attr_info['Minimum'], attr_info['Maximum']))
    #     return creator.Individual(individual)
    
    
    
    
    # 寻找最大GI的
    # def create_individual(self, input_ranges, toolbox):
    #     individual = []
    #     # 定义特定的整型键列表
    #     specific_int_keys = [
    #         'Additive_4', 'Turning times', 'Material_2', 'V1_Ventilation Type', 'V2_Ventilation Interval (min)',
    #         'Material_Main', 'Material_3', 'Composting Method', 'Additive_2', 'Aeration method', 'Additive_1',
    #         'Composting Method', 'Additive_3', 'M1_is Enclosed'
    #     ]
        
    #     for attr_name, attr_info in input_ranges.items():
    #         if attr_name in self.cat_feas:
    #             individual.append(toolbox.attr_int(attr_info['Minimum'], attr_info['Maximum']))
    #         elif attr_name in specific_int_keys:
    #             individual.append(toolbox.attr_int(attr_info['Minimum'], attr_info['Maximum']))
    #         else:
    #             individual.append(toolbox.attr_float(attr_info['Minimum'], attr_info['Maximum']))
        
    #     return creator.Individual(individual)



    # # 从最大GI中去相应相同的特征值  非键值对
    # def create_individual(self, input_ranges, toolbox):
    #     individual = []
    #     # 定义特定的整型键列表
    #     specific_int_keys = [
    #         'Additive_4', 'Turning times', 'Material_2', 'V1_Ventilation Type', 'V2_Ventilation Interval (min)',
    #         'Material_Main', 'Material_3', 'Composting Method', 'Additive_2', 'Aeration method', 'Additive_1',
    #         'Composting Method', 'Additive_3', 'M1_is Enclosed'
    #     ]
        
 
    #     # 读取CSV文件
    #     csv_path = 'output\Ga\GI__04\GaReasult\individuals_min_Final GI (%).csv'
    #     csv_data = pd.read_csv(csv_path)
        
    #     # 将 'Final GI (%)' 从列表转换为单个值
    #     max_row = csv_data.iloc[0]
    #     # print(max_row)

    #     for attr_name, attr_info in input_ranges.items():
    #         if attr_name in specific_value_keys:
    #             # 从最大值行中获取相应的特征值
    #             value = max_row[attr_name]
    #             if attr_name in specific_int_keys:
    #                 individual.append(int(value))  # 确保是整型
    #             else:
    #                 individual.append(float(value))  # 确保是浮点型
    #         elif attr_name in self.cat_feas:
    #             individual.append(toolbox.attr_int(attr_info['Minimum'], attr_info['Maximum']))
    #         elif attr_name in specific_int_keys:
    #             individual.append(toolbox.attr_int(attr_info['Minimum'], attr_info['Maximum']))
    #         else:
    #             individual.append(toolbox.attr_float(attr_info['Minimum'], attr_info['Maximum']))
    #     return creator.Individual(individual)


    # 从最大GI中去相应相同的特征值  键值对
    def create_individual(self, input_ranges, toolbox):
        individual = {}
        # # 定义特定的整型键列表
        # specific_int_keys = [
        #     'Additive_4', 'Turning times', 'Material_2', 'V1_Ventilation Type', 'V2_Ventilation Interval (min)',
        #     'Material_Main', 'Material_3', 'Composting Method', 'Additive_2', 'Aeration method', 'Additive_1',
        #     'Composting Method', 'Additive_3', 'M1_is Enclosed'
        # ]

        # # 定义特定的值键列表 取最大的Final GI的参数组合的共同部分
        # specific_value_keys = [
        #     'V3_Ventilation Duration (min)', 'Initial Moisture Content (%)', 'Period (d)', 'V1_Ventilation Type',
        #     'V4_Ventilation Day', 'Additive_2', 'V2_Ventilation Interval (min)', 'Compost volume (m3)',
        #     'Application Rate (%DW)', 'V5_Ventilation rate (L/min/kg iniDW)', 'Initial C/N (%)', 'Material_Main',
        #     'M1_is Enclosed', 'Turning times', 'Additive_1', 'Material_2'
        # ]
        
        
        if(self.target != "Final GI (%)"):
            # 读取CSV文件
            csv_path = 'output\\Ga\\GI__04\\GaReasult\\individuals_max_Final GI (%).csv'
            csv_data = pd.read_csv(csv_path)
            
            # 将 'Final GI (%)' 从列表转换为单个值
            max_row = csv_data.iloc[0]
            # print(max_row)

            for attr_name, attr_info in input_ranges.items():
                if attr_name in common_fixed:
                    # 从最大值行中获取相应的特征值
                    value = max_row[attr_name]
                    if attr_name in specific_int_keys:
                        individual[attr_name] = int(value)  # 确保是整型
                    else:
                        individual[attr_name] = float(value)  # 确保是浮点型
                elif attr_name in specific_int_keys:
                    individual[attr_name] = toolbox.attr_int(attr_info['Minimum'], attr_info['Maximum'])
                else:
                    individual[attr_name] = toolbox.attr_float(attr_info['Minimum'], attr_info['Maximum'])
        else:
            # 处理的是Final GI (%)max
            for attr_name, attr_info in input_ranges.items():
                if attr_name in self.cat_feas:
                    individual[attr_name] = toolbox.attr_int(attr_info['Minimum'], attr_info['Maximum'])
                elif attr_name in specific_int_keys:
                    individual[attr_name] = toolbox.attr_int(attr_info['Minimum'], attr_info['Maximum'])
                else:
                    individual[attr_name] = toolbox.attr_float(attr_info['Minimum'], attr_info['Maximum'])
        return creator.Individual(individual)
    
    # 最原始的不进行处理的
    # def evaluate(self, individual, input_ranges):
    #     individual_with_names = {attr_name: value for attr_name, value in zip(input_ranges.keys(), individual)}

    #     # 检查每个特征值是否超出范围，如果超出范围则返回一个非常大的适应度值
    #     for key, value in individual_with_names.items():
    #         if key in self.cat_feas:
    #             if not isinstance(value, int) or value < input_ranges[key]['Minimum'] or value > input_ranges[key]['Maximum']:
    #                 return (1e6,)  # 返回一个非常大的适应度值，表示不合法的个体
    #         else:
    #             if value < input_ranges[key]['Minimum'] or value > input_ranges[key]['Maximum']:
    #                 return (1e6,)  # 返回一个非常大的适应度值，表示不合法的个体
    #     # 使用模型进行预测
    #     individual_df = pd.DataFrame([individual_with_names], columns=input_ranges.keys())
    #     prediction = self.model.predict(individual_df)

    #     # 计算模型输出并返回其负值作为适应度（因为我们是最小化问题）
    #     fitness = -prediction
    #     return fitness,
    
    
    # 保证整型的测试代码
    def evaluate(self, individual, input_ranges):
        # print(individual)
        individual_with_names = {attr_name: value for attr_name, value in zip(input_ranges.keys(), individual.values())}
        # print(individual_with_names)
        # # 定义特定的整型键列表
        # specific_int_keys = [
        #     'Additive_4', 'Turning times', 'Material_2', 'V1_Ventilation Type', 'V2_Ventilation Interval (min)',
        #     'Material_Main', 'Material_3', 'Composting Method', 'Additive_2', 'Aeration method', 'Additive_1',
        #     'Composting Method', 'Additive_3', 'M1_is Enclosed'
        # ]
        
        # 检查每个特征值是否超出范围，如果超出范围则返回一个非常大的适应度值
        for key, value in individual_with_names.items():
            
            if key in self.cat_feas or key in specific_int_keys:
                if not isinstance(value, int) or value < input_ranges[key]['Minimum'] or value > input_ranges[key]['Maximum']:
                    if self.target == "Final GI (%)":
                        return (1e6,)  # 最大化问题，直接返回预测值作为适应度
                    else:
                        return (-1e6,)  # 最小化问题，返回预测值的负值作为适应度      
                     # 返回一个非常大的适应度值，表示不合法的个体
            else:
                if value < input_ranges[key]['Minimum'] or value > input_ranges[key]['Maximum']:
                    if self.target == "Final GI (%)":
                        return (1e6,)  # 最大化问题，直接返回预测值作为适应度
                    else:
                        return (-1e6,)  # 最小化问题，返回预测值的负值作为适应度   

        # 使用模型进行预测
        individual_df = pd.DataFrame([individual_with_names], columns=input_ranges.keys())
        prediction = self.model.predict(individual_df)
        # print(prediction)
        if self.target == "Final GI (%)":
            return (prediction,)  # 最大化问题，直接返回预测值作为适应度
        else:
            return (-prediction,)  # 最小化问题，返回预测值的负值作为适应度
    



    def optimization(self, target, population_size=200, n_generations=100, cxpb=0.5, mutpb=0.01, num_runs=200):
        # 创建输入范围字典和输出范围字典
        input_ranges = {key: value for key, value in self.min_max_values.items() if key != target}
        # output_range_1 = {key: value for key, value in self.min_max_values.items() if key == target}
        # # 创建属性名称到数值的映射
        # attribute_mapping = {key: idx for idx, key in enumerate(input_ranges.keys())}
        
        

        

        

        toolbox = base.Toolbox()
        
        if(self.target != "Final GI (%)"):
            # 最小化目标函数
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", dict, fitness=creator.FitnessMin)
        else:
            # 创建一个FitnessMax类，用于最大化目标函数
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", dict, fitness=creator.FitnessMax)
        
        # 定义生成整数属性的方法
        toolbox.register("attr_int", lambda minimum, maximum: random.randint(round(minimum), round(maximum)))
        # 定义生成浮点数属性的方法
        toolbox.register("attr_float", lambda minimum, maximum: random.uniform(minimum, maximum))
        # 注册个体生成方法
        toolbox.register("individual", self.create_individual, toolbox=toolbox, input_ranges=input_ranges)
        # 初始化种群
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate",  lambda ind: self.evaluate(ind, input_ranges=input_ranges))
        if(self.target != "Final GI (%)"):
             # 交叉、变异、选择   这是固定下面列表的版本
            """       
            # 定义特定的值键列表
            specific_value_keys = [
                'V3_Ventilation Duration (min)', 'Initial Moisture Content (%)', 'Period (d)', 'V1_Ventilation Type',
                'V4_Ventilation Day', 'Additive_2', 'V2_Ventilation Interval (min)', 'Compost volume (m3)',
                'Application Rate (%DW)', 'V5_Ventilation rate (L/min/kg iniDW)', 'Initial C/N (%)', 'Material_Main',
                'M1_is Enclosed', 'Turning times', 'Additive_1', 'Material_2'
            ]"""
            ### 注册工具
            toolbox.register("mate", custom_mate, specific_keys=common_fixed)  # 交叉方式使用自定义交叉
            toolbox.register("mutate", custom_mutate, specific_keys=common_fixed)  # 变异方式使用自定义变异
            toolbox.register("select", tools.selTournament, tournsize=3)   
        else:
            # 交叉、变异、选择   这是所有都变的版本
            # toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 交叉方式使用 Blend 交叉
            # toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)  # 变异方式使用高斯变异
            # toolbox.register("select", tools.selTournament, tournsize=3)
            ### 注册工具
            toolbox.register("mate", custom_mate, specific_keys=[])  # 交叉方式使用自定义交叉
            toolbox.register("mutate", custom_mutate, specific_keys=[])  # 变异方式使用自定义变异
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
            min_target = min(self.df[target])
        else:
            min_target = max(self.df[target])
        
        # GI要选大于百分之80以上的
        # min_target = min(self.df[target])
        for index, row in self.df.iterrows():
            if row[target] == min_target:
                min_combinations.append(row)


        if(self.target != "Final GI (%)"):
            print(f"\nCombinations with minimum {target} ({min_target}):")
            min_combinations_df = pd.DataFrame(min_combinations)
            min_combinations_df.to_csv(f'{self.save_path}individuals_min_{target}.csv', index=False)
        else:
            print(f"\nCombinations with maximum {target} ({min_target}):")
            min_combinations_df = pd.DataFrame(min_combinations)
            min_combinations_df.to_csv(f'{self.save_path}individuals_max_{target}.csv', index=False)
        print(f'{sys._getframe().f_code.co_name} finish')

