import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly_express as px
import plotly.graph_objects as go
import os
from data_utils.utils import Data_Pro
import math


# 配置文件
input_file = 'data\data_0625\堆肥数据库-机器学习v5.xlsx' # 文件输入路径
output_file = 'data\data_selected_0625_0.5\__19' # 文件输出路径
os.makedirs(output_file, exist_ok=True)
last_number = int(output_file.split('_')[-1])
use_norm = False # 是否使用归一化
use_mean_encodeing = False # 使用平均数编码
missing_rate = 0.5 # 筛选缺失值的比例阈值
fillcontent = 0
times = 3
sheet_name = "19"


# 这一块自己定义
geograpy_factors = ['Longitude (E)', 'Latitude (N)']  #  这一块内容没有

# 类别特征
catetory_factors = ['Material_Main', 'Material_2', 'Material_3', 'Additive Species', 'Additive_1','Additive_2', 
                    'Additive_3',"Additive_4",'Method', 'M1_isEnclosed', 'M2_isTurning', 'M3_isForce aeration',
                    'M4_isVessel', 'M5_isReactor', 'V1_Ventilation Type', 'V6_Extra']
# catetory_factors = ['Crop type']

# 数值特征
# numeric_factors = ['Application Rate (%DW)', 'Initial Moisture Content (%)', 'Initial pH','Initial TN (%)','Initial TC (%)',
# 'Initial C/N (%)', 'Initial EC (ms/cm)', 'Initial GI (%)', 'Initial FW (kg)', 'Initial DW (kg)', 'Initial NO2-N (g/kg)', 
# 'Initial NH3-N (g/kg)', 'Final Moisture Content (%)', 'Final pH', 'Final TN (%)', 'Final TC (%)', 'Final C/N (%)', 
# 'Final EC (ms/cm)', 'Final GI (%)', 'Final FW (kg)', 'Final DW (kg)', 'Final NH3-N (g/kg)', 'Final NO2-N (g/kg)', 
# 'Period (d)', 'Compost volume (m3)', 'Initial density (kg/L)', 'Turning times', 'V2_Ventilation Interval (min)', 
# 'V3_Ventilation Duration (min)', 'V4_Ventilation Day', 'V5_Ventilation rate (L/min/kg iniDW)']
numeric_factors = ['Application Rate (%DW)', 'Initial Moisture Content (%)', 'Initial pH','Initial TN (%)',
                   'Initial TC (%)','Initial C/N (%)', 'Initial EC (ms/cm)', 'Initial GI (%)', 'Initial FW (kg)', 
                   'Initial DW (kg)', 'Initial NO2-N (g/kg)', 'Initial NH3-N (g/kg)', 'Period (d)', 'Compost volume (m3)', 
                   'Initial density (kg/L)', 'Turning times', 'V2_Ventilation Interval (min)', 'V3_Ventilation Duration (min)',
                   'V4_Ventilation Day', 'V5_Ventilation rate (L/min/kg iniDW)']
# numeric_factors = ['Sand (%)', 'Silt (%)', 'Clay (%)', 'SOC (%)', 'TN (%)', 'C/N', 'pH', 'BD', 'CEC', 'N application', 'BNE', 'MAT (°C)', 'MAP (mm)']
target = ['TN loss (%)', 'NH3-N (g)', 'N2O-N (g)', 
          'NH3-N loss (%)', 'N2O-N loss (%)', 'TC loss (%)', 
          'CH4-C (g)', 'CO2-C (g)', 'CH4-C loss (%)', 'CO2-C loss (%)']
# 要预测的目标  16,18
if sheet_name not in ['分组2','分组4']:

    target = ['TN loss (%)','NH3-N loss (%)', 'N2O-N loss (%)', 'TC loss (%)', 
            'CH4-C loss (%)', 'CO2-C loss (%)']
else:
    target =  [
        "Final Moisture Content (%)", 
        "Final pH", 
        "Final TN (%)", 
        "Final TC (%)", 
        "Final C_N (%)", 
        "Final EC (ms_cm)", 
        "Final GI (%)", 
        "Final NH3-N (g_kg)", 
        "Final NO2-N (g_kg)"
    ]
    
# 12,13,14,15
# target = [
#             'TN loss (%)','NH3-N loss (%)', 'N2O-N loss (%)', 'TC loss (%)', 
#             'CH4-C loss (%)', 'CO2-C loss (%)',
#             "Final GI (%)",
#             ]

## 17  19
# target = [
#             "Final GI (%)",
#             ]


if last_number in [13, 15, 18, 19]:
    target = [
                "Final GI (%)",
                ]
elif last_number in [12, 14]:
    # 12  14 
    target = ['TN loss (%)','NH3-N loss (%)', 'N2O-N loss (%)', 'TC loss (%)', 
            'CH4-C loss (%)', 'CO2-C loss (%)',
            "Final GI (%)",
            ]
elif last_number in [16,17]:
    # 16 17
    target = ['TN loss (%)','NH3-N loss (%)', 'N2O-N loss (%)', 'TC loss (%)', 
            'CH4-C loss (%)', 'CO2-C loss (%)',
            ]

# target = ['TN loss (%)','NH3-N loss (%)', 'N2O-N loss (%)', 'TC loss (%)', 
#         'CH4-C loss (%)', 'CO2-C loss (%)']

# target = ['EF %', 'LN(EF)', 'LNR(N2O)', 'N2O rate(kg N ha-1 y-1)']

# col = catetory_factors + numeric_factors + target
# 尝试使用不同的编码格式读取文件
encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'gbk']

# # 去除所有单元格左右两边的空格

data_all, numeric_factors, catetory_factors = Data_Pro.read_and_split_columns(input_file,encodings,sheet_name=sheet_name)
numeric_factors = [col for col in numeric_factors if col not in target]
print("===========data_all的列===========")
print('num', numeric_factors)
print('non num', catetory_factors)

data_all = data_all.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
nan_list = ['nan', 'Nan', 'NaN', 'NAN', 'not specified']
data_all.replace('-', 0, inplace=True)
data_all.replace(nan_list, np.nan, inplace=True)

col = catetory_factors + numeric_factors + target
# for encoding in encodings:
#     try:
#         if 'csv' in input_file:
#             data_all = pd.read_csv(input_file,encoding=encoding)[col]
#         else:
#             data_all = pd.read_excel(input_file,sheet_name="预测分组1")[col]
#     except Exception as e:
#         print(f"An error occurred: {e}")




# 初步探查每个字段的含义
columns_all = data_all.columns.tolist()
print(columns_all)
print(len(columns_all))
if 'V2_Ventilation Interval (min)' in numeric_factors:
    data_all['V2_Ventilation Interval (min)'].unique()

    
input_cols = catetory_factors + numeric_factors
all_cols = input_cols + target
data_all = data_all[all_cols]
print(data_all.shape)
# 观察有没有异常值
data_all.describe()


# 将 'Initial pH' 列转换为数值类型，无法转换的字符串将变为 NaN
data_all['Initial pH'] = pd.to_numeric(data_all['Initial pH'], errors='coerce')

# 应用 clip 方法，将 'Initial pH' 列的值限制在 3 到 14 之间
data_all['Initial pH'] = data_all['Initial pH'].clip(lower=3, upper=14)

# 利用方差检验法检查异常值

data_all['Turning times'].unique()






# 查看数据缺失值
print("===========data_all的列===========")
print(data_all.columns)
missing = data_all.isnull().sum() / len(data_all)
print(missing)
missing = missing[missing>=0]
missing.sort_values(inplace=True)
# 筛去缺失值大于25%的数据列
columns_nomiss = missing[missing < missing_rate].index.tolist()

columns_miss = missing[missing >= missing_rate].index.tolist()
print("===========处理完===========")
print("columns_miss",columns_miss)
print(f'{len(columns_miss)} is missing')
print("columns_nomiss",columns_nomiss)
columns_nomiss = [i for i in columns_nomiss if i not in target]

# 更新类别特征和数值特征
catetory_factors = [col for col in catetory_factors if col in columns_nomiss]
numeric_factors = [col for col in numeric_factors if col in columns_nomiss]

print('catetory_factors update',catetory_factors)
print('numeric_factors update',numeric_factors)


data_target = data_all[target]
data_input = data_all[columns_nomiss]

input_cols = data_input.columns.tolist()

# 填充缺失值
data_input = Data_Pro.fill_missing(data_input, numeric_factors,catetory_factors,fillcontent)
print("===========填充缺失值之后===========")
print(data_input.columns)
missing = data_input.isnull().sum() / len(data_all)
print(missing)
if use_norm:
    data_all_norm = Data_Pro.normalize_features(data_input, numeric_factors)
else:
    data_all_norm = data_input

# 查看每一列的异常值数量
error_train = data_all_norm.copy()
print("error_train_ssssss" , error_train.columns)
error_train, _ = Data_Pro.remove_outliers_knn(error_train, numeric_factors, filled=True)

data_all_norm = error_train
# error_train = Data_Pro.find_outlines_by_3segama(error_train, fea , times)
# 或者，删除异常值
# print("error_train",error_train.columns)
# for fea in numeric_factors:
#     condition = error_train[fea + 'outlines'] == 'error'
#     data_all = data_all[~condition]

high_cat_features, low_cat_features = Data_Pro.high_or_low_category(data_input, catetory_factors)
print('low_cat_features',low_cat_features)
print('high_cat_features',high_cat_features)



# 直接保存以供预测
for trg in target:
    data_final = data_all_norm.copy()
    print("===========data_final的列有没有重复的===========")
    print(data_final.columns)
    final_col = input_cols
    print("===========final_col的列===========")
    print(final_col)
    if use_mean_encodeing:
        for hcf in high_cat_features:
            final_col = [col for col in final_col if hcf not in col and col != f'{hcf}_{trg}']
            data_final[hcf] = data_final[f'{hcf}_{trg}']
    print("===========data_final_use_mean_encodeing的列===========")
    print(data_final.columns)
    data_for_pre = pd.concat([data_final[final_col], data_target[trg]], axis=1)
    print("===========data_for_pre没有筛选异常值的列有没有重复的===========")
    print(data_for_pre.columns)
    data_for_pre = data_for_pre.dropna(subset=[trg])
    # 在这才去类别特征处理
    if use_mean_encodeing:
        # 目前还是不使用平均数编码（因为要么高基数的输入列缺的多，要么target缺的多）
        mapping_dict = Data_Pro.encode_categories(data_for_pre, low_cat_features)
        data_for_pre = Data_Pro.mean_encoding(pd.concat([data_for_pre, data_target]), high_cat_features, target)
    else:
        mapping_dict = Data_Pro.encode_categories(data_for_pre, catetory_factors)
        
    # 筛选掉目标异常值
    print('{} before: {}'.format(trg, len(data_for_pre)))
    data_for_pre, _ = Data_Pro.remove_outliers_knn(data_for_pre, [trg], n_neighbors=50)
    print("===========data_for_pre的列有没有重复的===========")
    print(data_for_pre.columns)
    print('{} after: {}'.format(trg, len(data_for_pre)))
    data_for_pre = data_for_pre.applymap(lambda x: round(x, 5) if isinstance(x, (int, float)) else x)
    data_for_pre.to_csv(f'{output_file}/data_for_{trg}.csv', index=False)


