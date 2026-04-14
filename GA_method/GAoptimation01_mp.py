import json
import joblib
import pandas as pd
from models import ModelBase, LGBTraining, CatTraining, RFTraining, GSRTraining, MLPTraining, XGBTraining, SVRTraining, LRTraining, RidgeTraining
from deap import base, creator, tools, algorithms
import random
import os
import sys
from utils import get_min_max_df, custom_mate,custom_mutate, main_NSGAII, get_union, evaluate_one
import multiprocessing
import asyncio

int_feas = ['Material_Main', 'Material_2', 'Material_3', 'Additive Species', 'Additive_1','Additive_2', 
                    'Additive_3',"Additive_4",'Method', 'M1_is Enclosed', 'M2_is Turning', 'M3_isForce aeration',
                    'M4_isVessel', 'M5_isReactor', 'V1_Ventilation Type', 'V6_Extra', 
                    'Aeration method', ' Composting Method']

float_feas = ['Application Rate (%DW)', 'Initial Moisture Content (%)', 'Initial pH','Initial TN (%)',
                   'Initial TC (%)','Initial C/N (%)', 'Initial EC (ms/cm)', 'Initial FW (kg)', 
                   'Initial DW (kg)', 'Initial NO2-N (g/kg)', 'Initial NH3-N (g/kg)', 'Compost volume (m3)', 
                   'Initial density (kg/L)', 'V3_Ventilation Duration (min)',
                 'V5_Ventilation rate (L/min/kg iniDW)']

int_feas_v1 = ['Period (d)', 'Turning times', 'V2_Ventilation Interval (min)', 'V4_Ventilation Day']

fix_fea = ['Material_Main']
fix_fea2 = ['Initial GI (%)'
            ]
# 打开JSON文件并读取数据
data_scale = {}
file = r"output/Ga/raw_useall_True/GaReasult/min_max_['Final GI (%)', 'NH3-N loss (%)', 'N2O-N loss (%)', 'CH4-C loss (%)', 'CO2-C loss (%)'].json"
with open(file, 'r', encoding='utf-8') as file:
    data_scale = json.load(file)
# 多目标优化
class NSGA_II:
    def __init__(self, model_names, model_paths, data_paths, output_file, targets, targets_cols, fix_nums):
        self.targets = targets
        # self.cat_feas = cat_feas
        self.fix_fea2_dict = {}
        self.model_names = model_names
        self.model_paths = model_paths
        self.output_file = output_file
        self.fixed_InGI  = {}
        
        # targets = ["Final GI (%)", 'NH3-N loss (%)', 'N2O-N loss (%)', 'CH4-C loss (%)', 'CO2-C loss (%)']
        self.df_raw = []
        for data_path in data_paths:
            df_tmp = pd.read_csv(data_path)
            self.df_raw.append(df_tmp)
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
        self.fix_nums = fix_nums
        #  targets = ["Final GI (%)", 'NH3-N loss (%)', 'N2O-N loss (%)', 'CH4-C loss (%)', 'CO2-C loss (%)']
        targets_tmp = ["Final GI (%)", 'NH3-N loss (%)', 'N2O-N loss (%)', 'CH4-C loss (%)', 'CO2-C loss (%)']
        for df_one,target in zip(self.df_raw,targets_tmp):
            for tmp_fix_one in fix_fea2:
                if tmp_fix_one in list(df_one.columns):
                    self.fix_fea2_dict[tmp_fix_one] = df_one[tmp_fix_one].mean()
                pass 

        os.makedirs(self.save_path, exist_ok=True)

    def get_model(self):
        self.model = []
        for model, model_path in zip(self.model_names, self.model_paths):
            modelClass = self.models_dict[model](is_bayesian=False)
            model_tmp = modelClass.model
            model_tmp = joblib.load(model_path)
            self.model.append(model_tmp)
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_min_max(self):
        # 存储所有输入特征和标签的最小值和最大值的字典
        min_max_values = {}


        for df_raw, target in zip(self.df_raw, self.targets):
            input_cols = list(set(list(df_raw.columns) + [target]))
            tmp = get_min_max_df(df_raw, input_cols, self.output_file, int_feas=int_feas, int_feas_v1=int_feas_v1, fix_feas=fix_fea2)
            min_max_values = get_union(min_max_values, tmp, int_feas) # 求并集
            min_max_json = json.dumps(min_max_values)
            with open(f'{self.output_file}/GaReasult/min_max_{target}.json', 'w') as f:
                f.write(min_max_json)

        min_max_json = json.dumps(min_max_values)
        with open(f'{self.output_file}/GaReasult/min_max_{self.targets}.json', 'w') as f:
            f.write(min_max_json)
            
        self.raw_dict = min_max_values
        for i, fea in enumerate(fix_fea):
            self.raw_dict[fea] = [self.fix_nums[i]]
        print(f'{sys._getframe().f_code.co_name} finish')
        return min_max_values
    
    @staticmethod
    def re_map(inidi):
        ref_mean_tbl = pd.read_csv("forecast_result/data/fore_data/ref_mean_tbl.csv")
        inidi["Initial TN (%)"] = ref_mean_tbl.loc[ref_mean_tbl['Material_Main'] == inidi["Material_Main"] , "Initial TN (%)"].values[0]
        inidi["Compost volume (m3)"] = ref_mean_tbl.loc[ref_mean_tbl['Material_Main'] == inidi["Material_Main"] , "Compost volume (m3)"].values[0]
        inidi["Additive_3"] = 4
        inidi["Additive_4"] = 1
        if inidi["Additive_1"] == 27:
            inidi["Additive_2"] = 11
        
        inidi["Initial TC (%)"] = inidi["Initial C/N (%)"] * inidi["Initial TN (%)"]
        if (inidi["Application Rate (%DW)"] > 10) or inidi["Application Rate (%DW)"] < 2:
            inidi["Application Rate (%DW)"] = random.uniform(2, 10)
        
        
        # 2、V4 ≤ Period, V4 >= Period/3
        # 如果 V4 大于 Period，则整个条件为 False
        condition = (inidi["V4_Ventilation Day"] < inidi["Period (d)"] // 3) or (inidi["V4_Ventilation Day"] > inidi["Period (d)"])
        if not condition:
            inidi["V4_Ventilation Day"] =  random.randint(inidi["Period (d)"]//3, inidi["Period (d)"])

        # 4、Reactor and M1 == 1
        # 如果 Reactor 为 False 或者 M1 不等于 1，则整个条件为 False
        condition = inidi[" Composting Method"] == 0
        if condition:
            inidi[" Composting Method"] = 0
            inidi["M1_is Enclosed"] = 1

        # 5、V1=None and V2=0 and V3=0 and V4=0 and V5=0
        # 只要 V1 不是 None 或者 V2、V3、V4、V5 中任何一个不等于 0，条件就为 False
        condition = inidi["V1_Ventilation Type"] == 2
        if condition:
            inidi["V1_Ventilation Type"] = 2 
            inidi["V2_Ventilation Interval (min)"] = 0
            inidi["V3_Ventilation Duration (min)"] = 0
            inidi["V4_Ventilation Day"] = 0
            inidi["V5_Ventilation rate (L/min/kg iniDW)"] = 0

        # 6、V1=continuous and V2=0 , V3=0
        # 如果 V1 不等于 'continuous' 或者 V2 不等于 0，则条件为 False
        condition = inidi["V1_Ventilation Type"] == 0
        if condition:
            inidi["V1_Ventilation Type"] = 0 
            inidi["V2_Ventilation Interval (min)"] = 0
            inidi["V3_Ventilation Duration (min)"] = 0

        # 7、V1=Intermittent and V2、V3 不同时为 0
        # 如果 V1 不等于 'Intermittent' 或者 V2 和 V3 同时为 0，则条件为 False
        condition = inidi["V1_Ventilation Type"] == 1
        if condition:
            inidi["V1_Ventilation Type"] = 1 
            if inidi["V2_Ventilation Interval (min)"] == 0:
                inidi["V2_Ventilation Interval (min)"] = random.randint(1, data_scale["V2_Ventilation Interval (min)"][1])
            if inidi["V3_Ventilation Duration (min)"] == 0: 
                 inidi["V3_Ventilation Duration (min)"] = random.randint(1, data_scale["V3_Ventilation Duration (min)"][1])
            
        # 9. 50 <= Intial Moisture Content <= 70
        condition = (50 >= inidi["Initial Moisture Content (%)"]) or (inidi["Initial Moisture Content (%)"] <= 70)
        if condition:
            inidi["Initial Moisture Content (%)"] = random.randint(50, 70)
            
        # 10. 5 <= Intial PH <= 8
        condition = (5 >= inidi["Initial pH"]) or (inidi["Initial pH"] <= 8)
        if condition:
            inidi["Initial pH"] = random.randint(5, 8)
            
        # 11. 10 <= Intial C/N <= 40
        condition = (10 >= inidi["Initial C/N (%)"]) or (inidi["Initial C/N (%)"] <= 40)
        if condition:
            inidi["Initial C/N (%)"] = random.randint(10, 40) 
            
        # 12. V5 <= 2
        condition = inidi["V5_Ventilation rate (L/min/kg iniDW)"] > 2
        if condition:
            inidi["V5_Ventilation rate (L/min/kg iniDW)"] = random.uniform(data_scale["V5_Ventilation rate (L/min/kg iniDW)"][0], 2)
                 
        return inidi
        

    # 从最大GI中去相应相同的特征值  键值对
    def create_individual(self, input_ranges, toolbox):
        individual = {}
        for attr_name, attr_info in input_ranges.items():
            if attr_name in int_feas:
                individual[attr_name] = toolbox.attr_int(attr_info)
            elif attr_name in int_feas_v1:
                try:
                    individual[attr_name] = toolbox.attr_int_v1(attr_info)
                except:
                    print(attr_name, attr_info)
            elif attr_name in fix_fea2:
                individual[attr_name] = toolbox.attr_fixed_fea2(attr_name)
            elif attr_name in float_feas:
                individual[attr_name] = toolbox.attr_float(attr_info)
            else:
                individual[attr_name] = toolbox.attr_int(attr_info)
        individual = NSGA_II.re_map(individual)
                
        return creator.Individual(individual)
    
    # 保证整型的测试代码
    def evaluate(self, individual, new_dict):
        predictions = []
        # for df_raw, model in zip(self.df_raw, self.model):
        #     in_range = {key: value for key, value in new_dict.items() if key not in self.targets and key in list(df_raw.columns)}
        #     tmp = evaluate_one(model, individual, in_range, int_feas)
        #     predictions.append(-tmp[0])
        
        for df_raw, model, target in zip(self.df_raw, self.model, self.targets):
            if target in self.targets_cols:
                cols = self.targets_cols[target]
            else:
                cols = list(df_raw.columns)
            in_range = {key: value for key, value in new_dict.items() if key not in self.targets and key in cols}
            tmp = evaluate_one(model, individual, in_range, int_feas, target, self.fixed_InGI)
            # if target == 'Final GI (%)':
            #     predictions.append(-tmp[0])
            # else:
            predictions.append(tmp[0])
                
        return tuple(predictions)
    
    async def run_optimization(self, input_ranges, i, targets, population_size=200, n_generations=100, cxpb=0.5, mutpb=0.01, num_runs=200, new_dict=None):
        # 创建输入范围字典和输出范围字典
        print(f'Run {i+1} begin..........')
        toolbox = base.Toolbox()
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0, -1.0, -1.0))
        creator.create("Individual", dict, fitness=creator.FitnessMulti)
        
        # 定义生成整数属性的方法
        def attr_int(lst: list):
            return random.sample(lst, 1)[0]
        def attr_float(lst: list):
            return random.uniform(lst[0], lst[1])
        def attr_int_v1(lst: list):
            return random.randint(int(lst[0]), int(lst[1]+1))
        def attr_fixed_fea2(fixed_name:str):
            return self.fix_fea2_dict[fixed_name]
        

        toolbox.register("attr_int", attr_int)
        toolbox.register("attr_float", attr_float)
        toolbox.register("attr_int_v1", attr_int_v1)
        toolbox.register("attr_fixed_fea2", attr_fixed_fea2)

        # 注册个体生成方法
        toolbox.register("individual", self.create_individual, toolbox=toolbox, input_ranges=input_ranges)
        # 初始化种群
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate",  lambda ind: self.evaluate(ind, new_dict))

        toolbox.register("mate", custom_mate, specific_keys=fix_fea)  # 交叉方式使用自定义交叉
        toolbox.register("mutate", custom_mutate, specific_keys=fix_fea)  # 变异方式使用自定义变异
        toolbox.register("select", tools.selTournament, tournsize=3)   
        
        # 开始训练 
        pop = main_NSGAII(toolbox, population_size, n_generations, cxpb, mutpb)
        best_individual = tools.selBest(pop, 1)[0]
        print(f"Run {i+1}: Best individual with fitness {best_individual.fitness.values}")
        best_individual_with_names = dict(best_individual)
        return best_individual, best_individual_with_names

    async def optimization(self, targets, population_size=200, n_generations=100, cxpb=0.5, mutpb=0.01, num_runs=200, new_dict=None):
        # 创建进程池
        # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        
        # 运行优化
        # print('开始优化。。。')
        # results = [pool.apply_async(self.run_optimization, (self, i, targets, population_size, n_generations, cxpb, mutpb, num_runs, new_dict)) for i in range(num_runs)]
        cols = []
        for df in self.df_raw:
            cols += list(df.columns)
        cols = list(set(cols))
        input_ranges = {key: value for key, value in new_dict.items() if key not in targets and key in cols}
        tasks = [asyncio.create_task(self.run_optimization(input_ranges, i, targets, population_size, n_generations, cxpb, mutpb, num_runs, new_dict)) for i in range(num_runs)]
        # print('完成优化。。。')
        
        # 获取结果
        results = await asyncio.gather(*tasks)
        best_individuals = results
        
        # 保存最优个体到CSV文件
        file_name = f"{self.save_path}individuals_{targets}_{self.fix_nums}.csv"
        data = []
        columns = list(input_ranges.keys()) + targets
        for best_individual, best_individual_with_names in best_individuals:
            row = [best_individual_with_names[attr] for attr in input_ranges.keys()]
            row.extend(best_individual.fitness.values)    
            data.append(row)

        self.df = pd.DataFrame(data, columns=columns)
        self.df.to_csv(file_name, index=False)
        print(f'{sys._getframe().f_code.co_name} finish')
        
    def do_optimization(self, targets, population_size=200, n_generations=100, cxpb=0.5, mutpb=0.01, num_runs=200, new_dict=None):
        asyncio.run(self.optimization(targets, population_size, n_generations, cxpb, mutpb, num_runs, new_dict))

        
    def get_optim_comb(self, target):
        min_combinations = []
        target_value = max(self.df[target])

        if(self.target != "Final GI (%)"):
            print(f"\nCombinations with minimum {target} ({target_value}):")
            min_combinations_df = pd.DataFrame(min_combinations)
            min_combinations_df.to_csv(f'{self.save_path}individuals_min_{target}.csv', index=False)
        else:
            print(f"\nCombinations with maximum {target} ({target_value}):")
            min_combinations_df = pd.DataFrame(min_combinations)
            min_combinations_df.to_csv(f'{self.save_path}individuals_max_{target}.csv', index=False)
        print(f'{sys._getframe().f_code.co_name} finish')
