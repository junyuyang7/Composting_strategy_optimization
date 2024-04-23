import subprocess
import os
import shutil
import json

# 环境的 Python 解释器路径
python_env_path = "F:/Anaconda/envs/duifei/python.exe"

input_file = 'data/data_new/4441vT_1.xlsx' # 文件输入路径
output_file = 'data/TN_NH3_N2O_TC loss_CH4-C loss_CO2-C loss' # 文件输出路径

# 这一块自己定义
geograpy_factors = ['Longitude (E)', 'Latitude (N)']

# 类别特征
catetory_factors = ['material_0', 'material_1','Excipients_1','Additive Species']
# catetory_factors = ['Crop type']

# 数值特征
numeric_factors = ['Application Rate (%)', 'initial moisture content(%)','initial pH','initial TN(%)','initial TC(%)','initial CN(%)']
# numeric_factors = ['Sand (%)', 'Silt (%)', 'Clay (%)', 'SOC (%)', 'TN (%)', 'C/N', 'pH', 'BD', 'CEC', 'N application', 'BNE', 'MAT (°C)', 'MAP (mm)']

# 要预测的目标
target = ['TN loss  (%)', 'NH3-N loss  (%)', "N2O-N loss  (%)",'TC loss  (%)','CH4-C loss (%)','CO2-C loss  (%)']
# target = ['EF %', 'LN(EF)', 'LNR(N2O)', 'N2O rate(kg N ha-1 y-1)']

data_pre = f"data_processing.ipynb"


# 将多个列表转换为字符串
geograpy_factors_str = json.dumps(geograpy_factors)
catetory_factors_str = json.dumps(catetory_factors)
numeric_factors_str = json.dumps(numeric_factors)
target_str = json.dumps(target)

# 构建命令
command = [
    python_env_path,
    "-m",
    "papermill",
    "数据分析.ipynb",
    data_pre,
    "-p", "input_file", input_file,
    "-p", "output_file", output_file,
    "-p", "geograpy_factors", geograpy_factors_str,
    "-p", "catetory_factors", catetory_factors_str,
    "-p", "numeric_factors", numeric_factors_str,
    "-p", "target", target_str,
]



# 执行命令
# subprocess.run(command)



# # 定义需要清空的文件夹列表
# folders_to_clear = [
#     "output/CH4-C loss (%)_shap",
#     "output/CO2-C loss  (%)_shap",
#     "output/N2O-N loss  (%)_shap",
#     "output/NH3-N loss  (%)_shap",
#     "output/TC loss  (%)_shap",
#     "output/TN loss  (%)_shap"
# ]

# # # 遍历每个文件夹
# # for folder in folders_to_clear:
# #     # 完整的文件夹路径，可以修改为绝对路径或根据实际情况调整
# #     folder_path = os.path.join(os.getcwd(), folder)
# #     if os.path.exists(folder_path):
# #         # 遍历文件夹中的所有文件和子文件夹
# #         for filename in os.listdir(folder_path):
# #             file_path = os.path.join(folder_path, filename)
# #             try:
# #                 if os.path.isfile(file_path) or os.path.islink(file_path):
# #                     os.unlink(file_path)
# #                 elif os.path.isdir(file_path):
# #                     shutil.rmtree(file_path)
# #             except Exception as e:
# #                 print(f'Failed to delete {file_path}. Reason: {e}')
# #     else:
# #         print(f"Folder not found: {folder_path}")

# 目标列表
targets = ['TN loss  (%)', 'NH3-N loss  (%)', "N2O-N loss  (%)", 'TC loss  (%)', 'CH4-C loss (%)', 'CO2-C loss  (%)']
# targets = ['CH4-C loss (%)']
# targets = ['TN loss  (%)', 'NH3-N loss  (%)', "N2O-N loss  (%)", 'TC loss  (%)', 'CO2-C loss  (%)']



# 环境的 Python 解释器路径
python_env_path = "F:/Anaconda/envs/SOCYIED/python.exe"
# 遍历每个目标
for target in targets:
    output_notebook = f"output_model_training_{target.replace(' ', '_').replace('(', '').replace(')', '')}.ipynb"
    command = [
        python_env_path,
        "-m",
        "papermill",
        "模型训练.ipynb",
        output_notebook,
        "-p", "target", target
    ]
    # 执行命令
    subprocess.run(command)



