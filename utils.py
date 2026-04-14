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
from collections import defaultdict
import sys
import warnings
# 忽略 VisibleDeprecationWarning 警告
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

int_feas_v1 = ['Period (d)', 'Turning times', 'V2_Ventilation Interval (min)', 'V4_Ventilation Day']
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
def get_data(data_path, seed, use_all=False):
    data_all_ef = pd.read_csv(data_path)
    X_all = data_all_ef.iloc[:, :-1]
    y_all = data_all_ef.iloc[:, -1]
    # 91
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=seed)
    y_train = y_train.reset_index(drop=True)
    input_cols = X_all.columns.tolist()
    if not use_all:
        return X_train, X_test, y_train, y_test, input_cols
    else:
        return X_all, None, y_all, None, input_cols
    
def get_data_select_col(data_path:str, seed, use_all=False, select_col=None):
    data_all_ef = pd.read_csv(data_path)
    # 获取最后一列的列名
    last_column_name = data_all_ef.columns[-1]

    # 获取启动当前进程的脚本文件名
    startup_script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

    # 生成保存文件的文件夹路径
    directory_path = os.path.join(os.path.dirname(data_path), startup_script_name)

    # 创建目录（如果不存在）
    os.makedirs(directory_path, exist_ok=True)

    # 生成保存文件的完整路径
    file_name = f"{last_column_name}_{len(select_col)}.csv"
    save_path = os.path.join(directory_path, file_name)
    data_all_ef[[*select_col, last_column_name]].to_csv(save_path, index=False)
    X_all = data_all_ef.iloc[:, :-1][select_col]
    y_all = data_all_ef.iloc[:, -1]
    # 82
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=seed)
    y_train = y_train.reset_index(drop=True)
    input_cols = X_all.columns.tolist()
    if not use_all:
        return X_train, X_test, y_train, y_test, input_cols
    else:
        return X_all, None, y_all, None, input_cols


def get_num_cols(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    # 动态获取第一个键
    first_key = next(iter(data))    
    # 提取 "rf" 部分的数据
    rf_data = data[first_key]

    # 找到值最大的 key
    max_key = max(rf_data, key=rf_data.get)

    print(f"The key with the maximum value is: {max_key}")
    return int(max_key)

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
def get_min_max_df(df: pd.DataFrame, cols: list, output_file: str=None, int_feas: list=None, int_feas_v1: list=None, fix_feas: list=None) -> dict:
    min_max_values = defaultdict(int)
    
    for feature in cols:
        if feature in fix_feas:
            min_max_values[feature] = [df[feature].mean()]
        elif feature not in int_feas:
            min_max_values[feature] = [round(df[feature].min(), 2), round(df[feature].max(), 2)]
        elif feature in int_feas_v1:
            min_max_values[feature] = [int(df[feature].min()), int(df[feature].max())]

        else:
            # 确保将数据作为 object 类型处理，且只存储一维数组或标量
            unique_values = df[feature].unique().tolist()
            if any(isinstance(i, (list, np.ndarray)) for i in unique_values):
                raise ValueError(f"Column {feature} contains nested sequences, which is not supported.")
            min_max_values[feature] = unique_values
    # print("------------min_max_values:{}--------------".format(min_max_values))
    # 将 min_max_values 转换为可序列化的类型
    min_max_values = convert_numpy_types(min_max_values)

    return min_max_values

    
def convert_numpy_types(obj):
    """将 numpy.int64 和 numpy.float64 类型转换为 Python 的内置类型"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    else:
        return obj


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

def main_NSGAII(toolbox, population_size=100, n_generations=50, cxpb=0.5, mutpb=0.2):
    toolbox.popSize = population_size
    toolbox.maxGen = n_generations
    toolbox.cxProb = cxpb
    toolbox.mutateProb = mutpb

    pop = toolbox.population(toolbox.popSize) # 父代
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    fronts = tools.emo.sortNondominated(pop, k=toolbox.popSize)
    # # 将每个个体的适应度设置为pareto前沿的次序
    # for idx, front in enumerate(fronts):
    #     for ind in front:
    #         ind.fitness.values = (idx+1, 0, 0, 0, 0)
    # # 创建子代
    # offspring = toolbox.selectGen1(pop, toolbox.popSize) # binary Tournament选择
    offspring = algorithms.varAnd(pop, toolbox, toolbox.cxProb, toolbox.mutateProb)

    # 第二代之后的迭代
    for gen in range(1, toolbox.maxGen):
        combinedPop = pop + offspring # 合并父代与子代
        # 评价族群
        fitnesses = toolbox.map(toolbox.evaluate, combinedPop)
        for ind, fit in zip(combinedPop,fitnesses):
            ind.fitness.values = fit
        # 快速非支配排序
        fronts = tools.emo.sortNondominated(combinedPop, k=toolbox.popSize, first_front_only=False)
        # 拥挤距离计算
        for front in fronts:
            tools.emo.assignCrowdingDist(front)
        # 环境选择 -- 精英保留
        pop = []
        for front in fronts:
            pop += front
        pop = toolbox.clone(pop)
        pop = tools.selNSGA2(pop, k=toolbox.popSize, nd='standard')
        # 创建子代
        offspring = toolbox.select(pop, toolbox.popSize)
        offspring = toolbox.clone(offspring)
        offspring = algorithms.varAnd(offspring, toolbox, toolbox.cxProb, toolbox.mutateProb)

    return pop

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

def get_inner_min_max(dict1: dict, dict2: dict, int_feas: list) -> dict:
    result_dict = {}
    # 遍历第一个字典
    for k, v in dict1.items():
        if k in dict2:
            if k not in int_feas:
                intersect_min = max(v[0], dict2[k][0])
                intersect_max = min(v[1], dict2[k][1])
                if intersect_min > intersect_max:
                    intersect_max = intersect_min + 0.1
                    print(f'{sys._getframe().f_code.co_name}中{k}有错误: 新字典和旧字典没有交集')
                result_dict[k] = [intersect_min, intersect_max]
            else:
                set1, set2 = set(v), set(dict2[k])
                inner_set = set1.intersection(set2)
                if len(inner_set) == 0:
                    inner_set = set2
                result_dict[k] = list(inner_set)
        else:
            result_dict[k] = v

    for k, v in dict2.items():
        if k not in result_dict:
            result_dict[k] = v
    return result_dict

def get_union(dict1, dict2, int_feas: list):
    result_dict = {}
    # 遍历第一个字典
    for k, v in dict1.items():
        if k in dict2:
            if k not in int_feas:
                intersect_min = min(v[0], dict2[k][0])
                intersect_max = max(v[1], dict2[k][1])
                result_dict[k] = [intersect_min, intersect_max]
            else:
                set1, set2 = set(v), set(dict2[k])
                inner_set = set1.union(set2)
                if len(inner_set) == 0:
                    inner_set = set2
                result_dict[k] = list(inner_set)
        else:
            result_dict[k] = v

    for k, v in dict2.items():
        if k not in result_dict:
            result_dict[k] = v
    return result_dict

def is_True(inidi):
    # 2、V4 ≤ Period
    # 如果 V4 大于 Period，则整个条件为 False
    condition_2 = True
    condition_3 = True
    condition_4 = True
    condition_5 = True
    condition_6 = True
    condition_7 = True
    condition_8 = True
    condition_9 = True
    condition_10 = True
    condition_11 = True
    condition_12 = True
    condition_13 = True
    condition_14 = True
    condition_15 = True
    ref_mean_tbl = pd.read_csv("forecast_result/data/fore_data/ref_mean_tbl.csv")
    if "Initial TN (%)" in inidi:
        condition_9 =  inidi["Initial TN (%)"] == ref_mean_tbl.loc[ref_mean_tbl['Material_Main'] == inidi["Material_Main"] , "Initial TN (%)"].values[0]
    if "Compost volume (m3)" in inidi:
        condition_10 =inidi["Compost volume (m3)"] == ref_mean_tbl.loc[ref_mean_tbl['Material_Main'] == inidi["Material_Main"] , "Compost volume (m3)"].values[0]
    if "Additive_3" in inidi:
        condition_11 = inidi["Additive_3"] == 4
    if "Additive_4" in inidi:
        condition_12 = inidi["Additive_4"] == 1
    if "Additive_1" in inidi and "Additive_2" in inidi and inidi["Additive_1"] == 27:
        condition_13 = inidi["Additive_2"] == 11
    if "Initial TC (%)" in inidi and "Initial C/N (%)" in inidi and "Initial TN (%)" in inidi:
        condition_14 = inidi["Initial TC (%)"] == inidi["Initial C/N (%)"] * inidi["Initial TN (%)"]
    if "Application Rate (%DW)" in inidi:
        condition_15 = inidi["Application Rate (%DW)"] <= 10 and inidi["Application Rate (%DW)"] >= 0
    condition_2 = inidi["V4_Ventilation Day"] <= inidi["Period (d)"]

    # 4、Reactor and M1 == 1
    # 如果 Reactor 为 False 或者 M1 不等于 1，则整个条件为 False
    condition = inidi[" Composting Method"] == 0
    if condition:
        condition_4 = inidi[" Composting Method"] == 0 and inidi["M1_is Enclosed"] == 1

    # 5、V1=None and V2=0 and V3=0 and V4=0 and V5=0
    # 只要 V1 不是 None 或者 V2、V3、V4、V5 中任何一个不等于 0，条件就为 False
    condition = inidi["V1_Ventilation Type"] == 2
    if condition:
        condition_5 = inidi["V1_Ventilation Type"] == 2 and any([inidi["V2_Ventilation Interval (min)"] == 0, inidi["V3_Ventilation Duration (min)"] == 0, inidi["V4_Ventilation Day"] == 0, inidi["V5_Ventilation rate (L/min/kg iniDW)"] == 0])

    # 6、V1=continuous and V2=0
    # 如果 V1 不等于 'continuous' 或者 V2 不等于 0，则条件为 False
    condition = inidi["V1_Ventilation Type"] == 0
    if condition:
        condition_6 = inidi["V1_Ventilation Type"] == 0 and inidi["V2_Ventilation Interval (min)"] == 0

    # 7、V1=Intermittent and V2、V3 不同时为 0
    # 如果 V1 不等于 'Intermittent' 或者 V2 和 V3 同时为 0，则条件为 False
    condition = inidi["V1_Ventilation Type"] == 1
    if condition:
        condition_7 = inidi["V1_Ventilation Type"] == 1 and not (inidi["V2_Ventilation Interval (min)"] == 0 or inidi["V3_Ventilation Duration (min)"] == 0)
    for key_one in int_feas_v1:    
        if not (inidi[key_one] % 1 == 0):
            condition_8 = False
            break
    return condition_2 and condition_4 and condition_5 and condition_6 and condition_7 and condition_8 and condition_9 and condition_10 and condition_11 and condition_12 and condition_13 and condition_14 and condition_15
    pass




def evaluate_one(model, individual, input_ranges, int_feas, target, fixed_always:dict):
    # print(individual)
    individual_with_names = {key: individual[key] for key in input_ranges.keys() if key in individual}
    # 检查每个特征值是否超出范围，如果超出范围则返回一个非常大的适应度值
    for i, (key, value) in enumerate(individual_with_names.items()):
        if not is_True(individual_with_names):
            return (-1e6,) if target == 'Final GI (%)' else (1e6,)
            pass
        elif(key in fixed_always.keys() and fixed_always[key] != value):
            return (-1e6,) if target == 'Final GI (%)' else (1e6,)
        elif key in int_feas:
            if value not in input_ranges[key]:
                return (-1e6,) if target == 'Final GI (%)' else (1e6,)
            pass
        else:
            try:
                if key != 'Initial TC (%)' and (value < input_ranges[key][0] or value > input_ranges[key][1]):
                    return (-1e6,) if target == 'Final GI (%)' else (1e6,)
            except:
                pass

    individual_df = pd.DataFrame([individual_with_names], columns=input_ranges.keys())
    prediction = model.predict(individual_df)

    return prediction