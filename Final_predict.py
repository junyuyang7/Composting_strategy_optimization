# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
import plotly_express as px
import plotly.graph_objects as go
import joblib
import os

pal=sns.color_palette("plasma_r", 70).as_hex()[2:]
temp = {
    'layout': go.Layout(font={'family': "Franklin Gothic", 'size': 12}, width=800)
}
colors=px.colors.qualitative.Plotly
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文

warnings.filterwarnings('ignore')

# %%
pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows', 50)

target = ['EF', 'LN(EF)'] # 预测的目标
# data_type = ['Maize', 'RICE', 'Vegetable', 'Wheat']
data_name = 'Wheat' # 输入数据名

data_path = f'data/data_new/升温1.5/{data_name}+1.5度.xlsx' # 输入数据路径

output_file = 'output/EF_LNEF_N2O_LNRN2O/'
save_file = f'{output_file}/升温1.5' # 输出数据文件夹
os.makedirs(save_file, exist_ok=True)

# %%
data_all = pd.read_excel(data_path)

# %%
data_all

# %%
geograpy_factors = ['Longitude', 'Latitude']
catetory_factors = ['Crop type']
numeric_factors = ['Sand', 'Silt(%)', 'Clay (%)', 'SOC (%)', 'TN (%)', 'C/N', 'pH', 'BD', 'CEC', 'N application', 'BNE', 'MAT (°C)', 'MAP (mm)']


# %%
data_all.replace(["nan", 'Nan', 'NaN', 'NAN', 'None', 'not specified'], np.nan, inplace=True)
input_cols = catetory_factors + numeric_factors
data = data_all[input_cols]
print(data.shape)
# 观察有没有异常值
data.describe()

# %%
# 利用方差检验法检查异常值
def find_outlines_by_3segama(df, fea):
    data_std = np.std(df[fea])
    data_mean = np.mean(df[fea])
    outlines_cut_off = data_std * 3
    lower_rule = data_mean - outlines_cut_off
    upper_rule = data_mean + outlines_cut_off
    # 新建列判断哪个具有异常值
    df[f'{fea}outlines'] = df[fea].apply(
        lambda x: 'error' if x > upper_rule or x < lower_rule else 'normal'
    )
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
    condition = error_train[f'{fea}outlines'] == 'error'
    mean_value = data[fea].mean()
    data.loc[condition, fea] = mean_value

# %% [markdown]
# ### 开始进行预测

# %%
for trg in target:
    save_path = save_file + f'/{data_name}_{trg}.xlsx'
    model_path = f'{output_file}/model_{trg}/lgb_model.pkl'

    from lightgbm import LGBMRegressor

    lgb_model = LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)

    lgb_model = joblib.load(model_path)
    y = lgb_model.predict(data, num_iteration=lgb_model.best_iteration_)

    data_all[trg] = y
    data_all.to_excel(save_path, index=False)
    data_all.head(10)

# %%



