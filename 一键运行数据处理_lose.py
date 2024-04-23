import subprocess
import os
import shutil
import json

import os

file_path = 'f:/Research/Composting_strategy_optimization/Composting_strategy_optimization/一键运行数据处理_lose.py'
print("Checking if file exists:", os.path.exists(file_path))

# 激活指定的环境
environment_name = "duifei"
activate_command = f"conda activate {environment_name}"
subprocess.run(activate_command, shell=True)

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
subprocess.run(command)



