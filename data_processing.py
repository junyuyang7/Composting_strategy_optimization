# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
import plotly_express as px
import plotly.graph_objects as go
import os

pal=sns.color_palette("plasma_r", 70).as_hex()[2:]
temp = {
    'layout': go.Layout(font={'family': "Franklin Gothic", 'size': 12}, width=800)
}
colors=px.colors.qualitative.Plotly
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows', 50)


# %%
input_file = 'data/构建模型的训练数据.xlsx' # 文件输入路径
output_file = 'data/EF_LNEF_N2O_LNRN2O' # 文件输出路径
os.makedirs(output_file, exist_ok=True)

use_norm = False # 是否使用归一化
use_mean_encodeing = False # 使用平均数编码
missing_rate = 0.7 # 筛选缺失值的比例阈值


# 这一块自己定义
geograpy_factors = ['Longitude (E)', 'Latitude (N)']

# 类别特征
# catetory_factors = ['material_0', 'material_1','Excipients','Additive Species']
catetory_factors = ['Crop type']

# 数值特征
# numeric_factors = ['Application Rate (%)','initial moisture content(%)','initial pH','initial TN(%)','initial TC(%)','initial CN(%)']
numeric_factors = ['Sand (%)', 'Silt (%)', 'Clay (%)', 'SOC (%)', 'TN (%)', 'C/N', 'pH', 'BD', 'CEC', 'N application', 'BNE', 'MAT (°C)', 'MAP (mm)']

# 要预测的目标
# target = ['TN loss (%)', 'NH3-N loss (%)', 'N2O-N loss (%)']
target = ['EF %', 'LN(EF)', 'LNR(N2O)', 'N2O rate(kg N ha-1 y-1)']

# %%
data_all = pd.read_excel(input_file)

# %%
# 初步探查每个字段的含义
columns_all = data_all.columns.tolist()
print(columns_all)
print(len(columns_all))

# %%
data_all.replace(["nan", 'Nan', 'NaN', 'NAN', 'None', 'not specified'], np.nan, inplace=True)
input_cols = catetory_factors + numeric_factors
all_cols = input_cols + target
data_all = data_all[all_cols]
print(data_all.shape)
# 观察有没有异常值
data_all.describe()

# %% [markdown]
# ### 检验异常值（三倍方差法）

# %%
# 利用方差检验法检查异常值
def find_outlines_by_3segama(df, fea):
    data_std = np.std(df[fea])
    data_mean = np.mean(df[fea])
    outlines_cut_off = data_std * 3
    lower_rule = data_mean - outlines_cut_off
    upper_rule = data_mean + outlines_cut_off
    # 新建列判断哪个具有异常值
    try:
        df[fea+'outlines'] = df[fea].apply(lambda x:str('error') if x>upper_rule or x<lower_rule else 'normal')
        # print('upper_rule:', upper_rule, '\nlower_rule', lower_rule, '\nfea', fea)
    except Exception as e:
        print('出现错误:', e)
        print('upper_rule:', upper_rule, '\nlower_rule', lower_rule, '\nfea', fea)
    
    return df

# %%
import math

# 查看每一列的异常值数量
error_train = data_all.copy()
# 计算要创建的子图网格的行数和列数
total_plots = len(numeric_factors)
rows = int(math.sqrt(total_plots))
cols = math.ceil(total_plots / rows)

# 创建一个包含所有子图的大图
fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
fig.tight_layout(pad=5.0)  # 调整子图之间的间距
check_fea = numeric_factors
for i, fea in enumerate(check_fea):
    row, col = i // cols, i % cols
    ax = axes[row, col]
    error_train = find_outlines_by_3segama(error_train, fea)
    error_train[f'{fea}outlines'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(f'{fea} Error')
    ax.set_xlabel(fea)
    ax.set_ylabel('Count')

# %%
# 将异常值填充为平均值
for fea in check_fea:
    condition = error_train[fea + 'outlines'] == 'error'
    mean_value = data_all[fea].mean()
    data_all.loc[condition, fea] = mean_value

# %%
data_all.describe()

# %%
data_all.info()

# %%
import pandas_profiling as pp
report = pp.ProfileReport(data_all)
report.to_file('report.html')

# %%
# 生成初步分析报告
report

# %% [markdown]
# ### 缺失值处理

# %%
# 查看数据缺失值
missing = data_all.isnull().sum() / len(data_all)
missing = missing[missing>0]
missing.sort_values(inplace=True)

fig=go.Figure()
fig.add_trace(go.Bar(x=missing.index, y=missing.values,
                    hovertemplate='%{x} Missing ratio = %{y:.2f}<extra></extra>'))
fig.update_layout(template=temp,title='Missing Ration', 
                  xaxis={'title':'Average Importance','zeroline': False},
                  yaxis_showgrid=False, margin={'l': 120,'t':80},
                  height=400, width=800)


# %%
# 筛去缺失值大于50%的数据列
columns_nomiss = missing[missing < missing_rate].index.tolist()
columns_miss = missing[missing >= missing_rate].index.tolist()
print(f'{len(columns_miss)} is missing')
print(columns_miss)
columns_nomiss = [i for i in columns_nomiss if i not in target]

# 更新类别特征和数值特征
catetory_factors = [col for col in catetory_factors if col in columns_nomiss]
numeric_factors = [col for col in numeric_factors if col in columns_nomiss]

print('catetory_factors update',catetory_factors)
print('numeric_factors update',numeric_factors)

# %%
columns_nomiss, len(columns_nomiss)

# %%
data_target = data_all[target]
data_input = data_all[columns_nomiss]

# %%
data_input.head(3)

# %%
data_target.head(3)

# %%
data_all.head(3)

# %%
# 填充缺失值
def fill_missing(df, numerical_fea):
    df[numerical_fea] = df[numerical_fea].fillna(-1)
    # df[numerical_fea] = df[numerical_fea].fillna(df[numerical_fea].mean())

input_cols = data_input.columns.tolist()
fill_missing(data_input, numeric_factors)

# %%
data_input.info()

# %%
data_input.head(3)

# %%
numeric_factors

# %% [markdown]
# #### 查看特征相关性

# %%
# 查看heatmap图
correlation_matrix = data_input[input_cols].corr(method='spearman')
f, ax = plt.subplots(figsize=(16,16))
plt.title('Correlation of Numeric Features',y=1,size=16)
sns.heatmap(correlation_matrix,square=True,ax=ax,annot=True)

# %%
input_cols + target

# %%
# 查看heatmap图
data_Xy = pd.concat([data_input, data_target], axis=1)

# 进行平均数编码的时候列名是{col}_{trg}，现在根据不同的trg，把列名弄成{col}, 创建副本

corr = data_Xy[input_cols + target].corr(method='spearman')
f, axes = plt.subplots(1, len(target), figsize=(16,12))
plt.title('Feature-Target Correlation Heatmap',y=1,size=16)
for i, trg in enumerate(target):
    sns.heatmap(corr[[trg]],square=True,ax=axes[i],annot=True)


# %% [markdown]
# #### 进行保存

# %%
data_input.info()

# %%
data_input.head(3)

# %%
# 归一化
def normalize_features(df, fea_columns):
    """
    基于均值和标准差进行归一化
    """
    for col in fea_columns:
        try:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
        except Exception as e:
            print('normalize_features出现错误：', e)
            print(f'{col}这一列有问题')
    return df

if use_norm:
    data_all_norm = normalize_features(data_input, numeric_factors)
else:
    data_all_norm = data_input

# %%
#### 处理类别特征（类别数量少的直接使用序列编码，类别数量太大>20使用平均数编码）
from sklearn.preprocessing import LabelEncoder

# 按类别数量区分高类别和低类别特征
def high_or_low_category(df, columns):
    high_cat_features = []
    low_cat_features = []

    for col in columns:
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            # 类别特征大于20认为是高基数类别特征
            if unique_count >= 20: high_cat_features.append(col)
            else: low_cat_features.append(col)
    
    return high_cat_features, low_cat_features

# 序列编码
def encode_categories(df, columns):
    dic_lst = [] # 返回映射关系的字典
    for col in columns:
        label_encoder = LabelEncoder()
        # 对类别数据进行编码
        encoded_column = label_encoder.fit_transform(df[col])
        df[col] = encoded_column
        # 返回类别到数字的映射字典
        category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        dic_lst.append(category_mapping)
    
    return dic_lst

# 平均数编码
def mean_encoding(df, cat_feature, target_feature):
    for cat in cat_feature:
        for trg in target_feature:
            try:
                # print(df.columns)
                df_nonan = df.dropna(subset=[cat, trg])
                display(df_nonan)
                mean_encoded_values = df_nonan.groupby(cat)[trg].mean()
                print(mean_encoded_values)
                df[f'{cat}_{trg}'] = df[cat].map(mean_encoded_values)
            except Exception as e:
                print('mean_encoding出现错误：', e)
                print('cat:', cat, 'trg:', trg)
    cols = [col for col in df.columns if col not in target_feature]
    return df[cols]

high_cat_features, low_cat_features = high_or_low_category(data_input, catetory_factors)
print('low_cat_features',low_cat_features)
print('high_cat_features',high_cat_features)


if use_mean_encodeing:
    # 是否使用平均数编码
    mapping_dict = encode_categories(data_input, low_cat_features)
    data_input = mean_encoding(pd.concat([data_input, data_target]), high_cat_features, target)
else:
    mapping_dict = encode_categories(data_input, catetory_factors)
data_input.head(3)

# %%
# 直接保存以供预测
for trg in target:
    data_final = data_all_norm.copy()
    final_col = input_cols
    if use_mean_encodeing:
        for hcf in high_cat_features:
            final_col = [col for col in final_col if hcf not in col and col != f'{hcf}_{trg}']
            data_final[hcf] = data_final[f'{hcf}_{trg}']
    data_for_pre = pd.concat([data_final[final_col], data_target[trg]], axis=1)
    data_for_pre = data_for_pre.dropna(subset=[trg])
    # 在这才去类别特征处理
    if use_mean_encodeing:
        # 目前还是不使用平均数编码（因为要么高基数的输入列缺的多，要么target缺的多）
        mapping_dict = encode_categories(data_for_pre, low_cat_features)
        data_for_pre = mean_encoding(pd.concat([data_for_pre, data_target]), high_cat_features, target)
    else:
        mapping_dict = encode_categories(data_for_pre, catetory_factors)
    data_for_pre.to_csv(f'{output_file}/data_for_{trg}.csv', index=False)


# %%



