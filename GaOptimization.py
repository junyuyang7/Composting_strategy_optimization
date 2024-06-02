from GA_method.GAoptimation import GAOptimation
from utils import get_best_model_name

output_file = 'output/TN_NH3_N2O_0531' # 输出文件夹
input_file = 'data/TN_NH3_N2O_0531' # 输入数据文件夹

targets = ['TN loss (%)', 'NH3-N (g)', 'N2O-N (g)', 'NH3-N loss (%)', 'N2O-N loss (%)', 'TC loss (%)', 'CH4-C (g)', 'CO2-C (g)', 'CH4-C loss (%)', 'CO2-C loss (%)']

# 设置参数
population_size = 200
n_generations = 100
cxpb = 0.5  # 交叉概率
mutpb = 0.01  # 变异概率
num_runs = 100  # 运行次数

for target in targets:
    print(f'--------------Optimizate {target}------------------')
    json_path = f'{output_file}/model_{target}/result_r2_{target}.json'
    data_path = f'{input_file}/data_for_{target}.csv'
    model_name = get_best_model_name(json_path)
    print(f'best_model is {model_name}')
    model_path = f'{output_file}/model_{target}/{model_name}/{model_name}_model.pkl'
    Ga_client = GAOptimation(model_name, model_path, data_path, output_file)

    Ga_client.get_model()
    Ga_client.get_min_max()
    Ga_client.optimization(target, 
                           population_size=population_size, 
                           n_generations=n_generations, 
                           cxpb=cxpb, 
                           mutpb=mutpb, 
                           num_runs=num_runs)
    Ga_client.get_optim_comb(target)
