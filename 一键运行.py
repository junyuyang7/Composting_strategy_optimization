import subprocess
import os
import shutil

# 定义需要清空的文件夹列表
folders_to_clear = [
    "output/CH4-C loss (%)_shap",
    "output/CO2-C loss  (%)_shap",
    "output/N2O-N loss  (%)_shap",
    "output/NH3-N loss  (%)_shap",
    "output/TC loss  (%)_shap",
    "output/TN loss  (%)_shap"
]

# # 遍历每个文件夹
# for folder in folders_to_clear:
#     # 完整的文件夹路径，可以修改为绝对路径或根据实际情况调整
#     folder_path = os.path.join(os.getcwd(), folder)
#     if os.path.exists(folder_path):
#         # 遍历文件夹中的所有文件和子文件夹
#         for filename in os.listdir(folder_path):
#             file_path = os.path.join(folder_path, filename)
#             try:
#                 if os.path.isfile(file_path) or os.path.islink(file_path):
#                     os.unlink(file_path)
#                 elif os.path.isdir(file_path):
#                     shutil.rmtree(file_path)
#             except Exception as e:
#                 print(f'Failed to delete {file_path}. Reason: {e}')
#     else:
#         print(f"Folder not found: {folder_path}")

# 目标列表
targets = ['TN loss  (%)', 'NH3-N loss  (%)', "N2O-N loss  (%)", 'TC loss  (%)', 'CH4-C loss (%)', 'CO2-C loss  (%)']
# targets = ['CH4-C loss (%)']
# targets = ['TN loss  (%)', 'NH3-N loss  (%)', "N2O-N loss  (%)", 'TC loss  (%)', 'CO2-C loss  (%)']


# 遍历每个目标
for target in targets:
    output_notebook = f"output_model_training_{target.replace(' ', '_').replace('(', '').replace(')', '')}.ipynb"
    command = [
        "papermill",
        "模型训练.ipynb",
        output_notebook,
        "-p", "target", target
    ]
    # 执行命令
    subprocess.run(command)
