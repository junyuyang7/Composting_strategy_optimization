import json
import joblib
import pandas as pd
from models import ModelBase, LGBTraining, CatTraining, RFTraining, GSRTraining, MLPTraining, XGBTraining, SVRTraining, LRTraining, RidgeTraining
from deap import base, creator, tools, algorithms
import random
import os
import sys
from utils import create_individual, evaluate, main, get_min_max_df
from functools import partial

cat_feas = ['Material_Main', 'Material_2', 'Material_3', 'Additive Species', 'Additive_1','Additive_2', 'Additive_3', 'Method', 'M1_isEnclosed', 'M2_isTurning', 'M3_isForce aeration', 'M4_isVessel', 'M5_isReactor', 'V1_Ventilation Type', 'V6_Extra']

class GAOptimation:
    def __init__(self, model_name, model_path, data_path, output_file):
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
        self.min_max_values = get_min_max_df(self.df_raw, input_cols)
        print(f'{sys._getframe().f_code.co_name} finish')

    def create_individual(self, input_ranges, toolbox):
        individual = []
        for attr_name, attr_info in input_ranges.items():
            if attr_name in self.cat_feas:
                individual.append(toolbox.attr_int(attr_info['Minimum'], attr_info['Maximum']))
            else:
                individual.append(toolbox.attr_float(attr_info['Minimum'], attr_info['Maximum']))
        return creator.Individual(individual)

    def evaluate(self, individual, input_ranges):
        individual_with_names = {attr_name: value for attr_name, value in zip(input_ranges.keys(), individual)}

        # 检查每个特征值是否超出范围，如果超出范围则返回一个非常大的适应度值
        for key, value in individual_with_names.items():
            if key in self.cat_feas:
                if not isinstance(value, int) or value < input_ranges[key]['Minimum'] or value > input_ranges[key]['Maximum']:
                    return (1e6,)  # 返回一个非常大的适应度值，表示不合法的个体
            else:
                if value < input_ranges[key]['Minimum'] or value > input_ranges[key]['Maximum']:
                    return (1e6,)  # 返回一个非常大的适应度值，表示不合法的个体
        # 使用模型进行预测
        individual_df = pd.DataFrame([individual_with_names], columns=input_ranges.keys())
        prediction = self.model.predict(individual_df)

        # 计算模型输出并返回其负值作为适应度（因为我们是最小化问题）
        fitness = -prediction
        return fitness,

    def optimization(self, target, population_size=200, n_generations=100, cxpb=0.5, mutpb=0.01, num_runs=200):
        # 创建输入范围字典和输出范围字典
        input_ranges = {key: value for key, value in self.min_max_values.items() if key != target}
        # output_range_1 = {key: value for key, value in self.min_max_values.items() if key == target}
        # # 创建属性名称到数值的映射
        # attribute_mapping = {key: idx for idx, key in enumerate(input_ranges.keys())}
        # 最小化目标函数
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        # 定义生成整数属性的方法
        toolbox.register("attr_int", lambda minimum, maximum: random.randint(round(minimum), round(maximum)))
        # 定义生成浮点数属性的方法
        toolbox.register("attr_float", lambda minimum, maximum: random.uniform(minimum, maximum))
        # 注册个体生成方法
        toolbox.register("individual", self.create_individual, toolbox=toolbox, input_ranges=input_ranges)
        # 初始化种群
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate",  lambda ind: self.evaluate(ind, input_ranges=input_ranges))
        
        # 交叉、变异、选择
        toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 交叉方式使用 Blend 交叉
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)  # 变异方式使用高斯变异
        toolbox.register("select", tools.selTournament, tournsize=3)

        # 开始训练
        best_individuals = []
        for i in range(num_runs):
            pop, stats, hof = main(toolbox, population_size, n_generations, cxpb, mutpb)
            best_individual = tools.selBest(pop, 1)[0]
            print(f"Run {i+1}: Best individual with fitness {best_individual.fitness.values[0]}")
            best_individual_with_names = {list(input_ranges.keys())[idx]: value for idx, value in enumerate(best_individual)}
            best_individuals.append((best_individual, best_individual_with_names))

        # 保存最优个体到CSV文件
        file_name = f"{self.save_path}individuals_{target}.csv"
        data = []
        columns = list(input_ranges.keys()) + [target]
        for best_individual, best_individual_with_names in best_individuals:
            row = [best_individual_with_names[attr] for attr in input_ranges.keys()]
            row.append(-best_individual.fitness.values[0])  # 添加适应度值
            data.append(row)

        self.df = pd.DataFrame(data, columns=columns)
        self.df.to_csv(file_name, index=False)
        print(f'{sys._getframe().f_code.co_name} finish')

    def get_optim_comb(self, target):
        min_combinations = []
        min_target = min(self.df[target])
        for index, row in self.df.iterrows():
            if row[target] == min_target:
                min_combinations.append(row)

        print(f"\nCombinations with minimum {target} ({min_target}):")
        min_combinations_df = pd.DataFrame(min_combinations)
        min_combinations_df.to_csv(f'{self.save_path}individuals_min_{target}.csv', index=False)
        print(f'{sys._getframe().f_code.co_name} finish')
