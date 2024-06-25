# 设置日志
import logging
import numpy as np
import random
import os
import plotly.express as px
import plotly.io as pio
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from deap import base, creator, tools, algorithms

def set_logger(log_path):
    logger = logging.getLogger()
    # logger.addFilter(lambda record: "findfont" not in record.getMessage())
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_path) 
    file_handler.setFormatter(logging.Formatter("%(asctime)s: %(levelname)s: %(message)s"))
    file_handler.addFilter(lambda record: "findfont" not in record.getMessage())
    logger.addHandler(file_handler)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# 设置各个储存路径
def set_filename(target, output_file, input_file):
    model_save_file = f'{output_file}/model_{target}' # 模型参数文件夹
    
    os.makedirs(model_save_file, exist_ok=True)
    data_path = f'{input_file}/data_for_{target}.csv'
    model_performance_path = f'{model_save_file}/Model_{target}.html'
    mse_json = f'{model_save_file}/result_mse_{target}.json'
    mae_json = f'{model_save_file}/result_mae_{target}.json'
    r2_json = f'{model_save_file}/result_r2_{target}.json'

    return data_path, model_performance_path, mse_json, mae_json, r2_json, model_save_file

# 获取数据
def get_data(data_path, seed):
    data_all_ef = pd.read_csv(data_path)
    X_all = data_all_ef.iloc[:, :-1]
    y_all = data_all_ef.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=seed)
    y_train = y_train.reset_index(drop=True)
    input_cols = X_all.columns.tolist()
    return X_train, X_test, y_train, y_test, input_cols

# 获得各个模型的效果对比条形图
def get_r2_compare(target, output_file, input_file):
    data_path, model_performance_path, mse_json, mae_json, r2_json, model_save_file = set_filename(target, output_file, input_file)
    with open(r2_json, 'r') as json_file:
        result_mse = json.load(json_file)

    result_mse = dict(sorted(result_mse.items(), key=lambda item: item[1]))
    categories = list(result_mse.keys())
    values = list(result_mse.values())
        
    # 创建柱状图
    # fig = px.bar(x=categories, y=values, title=f'Models Performance in {target}', color=color_mapping)
    fig = px.bar(x=categories, y=values, title=f'Models Performance in {target}')
    fig.update_layout(template="seaborn")
    # 显示图表
    # fig.show()
    # 保存柱状图为 HTML 文件
    pio.write_html(fig, file=model_performance_path)

# GA
def get_min_max_df(df, cols, output_file):
    min_max_values = {}
    for feature in cols:
        # 如果特征不存在，创建新的特征项
        if feature not in min_max_values:
            min_max_values[feature] = {'Minimum': None, 'Maximum': None}
        # 计算输入特征的最小值和最大值，并更新字典中的值
        if min_max_values[feature]['Minimum'] is None:
            min_max_values[feature]['Minimum'] = round(df[feature].min(), 2)
        else:
            min_max_values[feature]['Minimum'] = min(round(min_max_values[feature]['Minimum'], 2), round(df[feature].min(), 2))
        if min_max_values[feature]['Maximum'] is None:
            min_max_values[feature]['Maximum'] = round(df[feature].max(), 2)
        else:
            min_max_values[feature]['Maximum'] = max(round(min_max_values[feature]['Maximum'], 2), round(df[feature].max(), 2))
    
    min_max_df = pd.DataFrame(min_max_values)
    min_max_df = min_max_df.transpose()
    min_max_df.to_csv(f'{output_file}/GaReasult/min_max.csv', index=False)

    return min_max_values

def create_individual(input_ranges, toolbox, cat_feas):
    individual = []
    for attr_name, attr_info in input_ranges.items():
        if attr_name in cat_feas:
            individual.append(toolbox.attr_int(attr_info['Minimum'], attr_info['Maximum']))
        else:
            individual.append(toolbox.attr_float(attr_info['Minimum'], attr_info['Maximum']))
    return creator.Individual(individual)

# 使用相应预测模型定义评估函数
def evaluate(individual, input_ranges, cat_feas, model):
    individual_with_names = {attr_name: value for attr_name, value in zip(input_ranges.keys(), individual)}

    # 检查每个特征值是否超出范围，如果超出范围则返回一个非常大的适应度值
    for key, value in individual_with_names.items():
        if key in cat_feas:
            if not isinstance(value, int) or value < input_ranges[key]['Minimum'] or value > input_ranges[key]['Maximum']:
                return (1e6,)  # 返回一个非常大的适应度值，表示不合法的个体
        else:
            if value < input_ranges[key]['Minimum'] or value > input_ranges[key]['Maximum']:
                return (1e6,)  # 返回一个非常大的适应度值，表示不合法的个体
    # 使用模型进行预测
    prediction = model.predict([individual])

    # 计算模型输出并返回其负值作为适应度（因为我们是最小化问题）
    fitness = -prediction
    return fitness,

# 进化算法
def main(toolbox, target ,population_size=100, n_generations=50, cxpb=0.5, mutpb=0.2):
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    if target != "Final GI (%)":
        stats.register("min", np.min)
    else:
        stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=n_generations, stats=stats, halloffame=hof, verbose=False)

    return pop, stats, hof

def get_best_model_name(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        max_value = float('-inf')
        data = json.load(file)
        # 查找最大值的键
        for key, value in data.items():
            if value > max_value:
                best_model_name = key
                max_value = value
    return best_model_name


# 自定义交叉函数
def custom_mate(ind1, ind2, specific_keys):
    # 保存特定键的原始值
    original_values1 = {key: ind1[key] for key in specific_keys}
    original_values2 = {key: ind2[key] for key in specific_keys}
    # 确保所有值为浮点数
    ind1_vals = [float(value) for value in ind1.values()]
    ind2_vals = [float(value) for value in ind2.values()]

    # 进行交叉操作
    ind1_vals, ind2_vals = tools.cxBlend(ind1_vals, ind2_vals, alpha=0.5)

    # 更新个体的值
    ind1.update(zip(ind1.keys(), ind1_vals))
    ind2.update(zip(ind2.keys(), ind2_vals))

    # 恢复特定键的原始值
    for key in specific_keys:
        ind1[key] = original_values1[key]
        ind2[key] = original_values2[key]

    return ind1, ind2

# 自定义变异函数
def custom_mutate(ind, specific_keys):
    # 保存特定键的原始值
    original_values = {key: ind[key] for key in specific_keys}

    # 确保所有值为浮点数
    ind_values = [float(value) for value in ind.values()]

    # 进行变异操作
    ind_values, = tools.mutGaussian(ind_values, mu=0, sigma=1, indpb=0.1)

    # 更新个体的值
    ind.update(zip(ind.keys(), ind_values))

    # 恢复特定键的原始值
    for key in specific_keys:
        ind[key] = original_values[key]

    return ind,