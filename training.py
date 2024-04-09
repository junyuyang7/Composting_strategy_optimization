# %%
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib
import os

import warnings
warnings.filterwarnings('ignore')  
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文  

# 设置Seaborn样式
sns.set(font='Microsoft YaHei')

# %%
# 获取字典保存各个模型的最终结果
result_model = {}

# target = ['TN loss (%)', 'NH3-N loss (%)', 'N2O-N loss (%)']
# target = ['EF %', 'LN(EF)', 'LNR(N2O)', 'N2O rate(kg N ha-1 y-1)']
target = 'N2O-N loss (%)' # 要预测啥就换个名字

output_file = 'output/TN_NH3_N2O' # 输出文件夹
input_file = 'data/TN_NH3_N2O' # 输入数据文件夹
model_save_file = f'{output_file}/model_{target}' # 模型参数文件夹

# output_file = 'output/EF_LNEF_N2O_LNRN2O'
# input_file = 'data/EF_LNEF_N2O_LNRN2O'
# model_save_file = f'output/EF_LNEF_N2O_LNRN2O/model_{target}'

os.makedirs(output_file, exist_ok=True)
os.makedirs(model_save_file, exist_ok=True)

# %%
data_path = f'{input_file}/data_for_{target}.csv'
model_performance_path = f'{output_file}/Model_{target}.html'
model_performance_json = f'{output_file}/result_model_{target}.json'

# %% [markdown]
# #### 数据集划分

# %%
data_all_ef = pd.read_csv(data_path)
X_all = data_all_ef.iloc[:, :-1]
y_all = data_all_ef.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=2023)
y_train = y_train.reset_index(drop=True)

## 为了正确评估模型性能，将数据划分为训练集和测试集，并在训练集上训练模型，在测试集上验证模型性能。
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True)

input_cols = X_all.columns.tolist()

# %%
X_train.info()

# %%
X_train.describe()

# %%
X_train

# %%
# 看是否有缺失值
y_train.isnull().any()

# %%
# 看是否有缺失值
print(X_train.isnull().any())

# %%
np.isnan(X_train).any()

# %% [markdown]
# ### 树模型预测（RF，XGB，LGB，CTB）

# %% [markdown]
# #### RF

# %%
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=1000, criterion='mse',
                            max_depth=10, min_samples_split=2,
                            min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                            max_features='auto',  max_leaf_nodes=None,
                            bootstrap=True, oob_score=False,
                            n_jobs=1, random_state=None,
                            verbose=0, warm_start=False)

rf_preds = np.zeros(X_train.shape[0])

for train_index, valid_index in kf.split(X_train):
    x_tr, x_va = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_tr, y_va = y_train.iloc[train_index], y_train.iloc[valid_index]
    
    print(x_tr.shape, y_tr.shape)
    rf_model.fit(x_tr, y_tr)
    rf_pred_valid = rf_model.predict(x_va)
    rf_preds[valid_index] = rf_pred_valid

    print('The rmse of the RFRegression is:',metrics.mean_squared_error(y_va,rf_pred_valid))
    x = list(range(0, len(y_va)))
    plt.figure(figsize=(40,10))
    plt.plot(x, y_va, 'g', label='ground-turth')
    plt.plot(x, rf_pred_valid, 'r', label='xgb predict')
    plt.legend(loc='best')
    plt.title('xgb-yield')
    plt.xlabel('sample')
    plt.ylabel('Yield_lnRR')
    plt.show()

# %%
rf_pred_test = rf_model.predict(X_test)
print('The rmse of the test is:',metrics.mean_squared_error(y_test,rf_pred_test))
x = list(range(0, len(y_test)))
plt.figure(figsize=(40,10))
plt.plot(x, y_test, 'g', label='ground-turth')
plt.plot(x, rf_pred_test, 'r', label='rf predict')
plt.legend(loc='best')
plt.title('rf-yield')
plt.xlabel('sample')
plt.ylabel('Yield_lnRR')
plt.show()

joblib.dump(rf_model, f'{model_save_file}/rf_model.pkl') 
result_model['RandomForest(k)'] = metrics.mean_squared_error(y_test,rf_pred_test)

df_tmp = pd.DataFrame({'pred': rf_pred_test, 'true': y_test})
df_tmp.to_csv(f'{model_save_file}/rf_pred.csv', index=False)

# %% [markdown]
# #### XGB

# %%
import xgboost as xgb

xgb_preds = np.zeros(X_train.shape[0])
kf = KFold(n_splits=num_folds, shuffle=True)
xgb_model = xgb.XGBRegressor(max_depth=8,
                        learning_rate=0.1,
                        n_estimators=300,
                        n_jobs=4,
                        colsample_bytree=0.8,
                        subsample=0.8,
                        random_state=32,
                        tree_method='hist'
                        )

for train_index, valid_index in kf.split(X_train):
    x_tr, x_va = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_tr, y_va = y_train.iloc[train_index], y_train.iloc[valid_index]
    
    print(x_tr.shape, y_tr.shape)
    xgb_model.fit(x_tr, y_tr)
    xgb_pred_valid = xgb_model.predict(x_va)
    xgb_preds[valid_index] = xgb_pred_valid

    # print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_valid,xgb_pred_valid))
    print('The rmse of the XgbRegression is:',metrics.mean_squared_error(y_va,xgb_pred_valid))
    x = list(range(0, len(y_va)))
    plt.figure(figsize=(40,10))
    plt.plot(x, y_va, 'g', label='ground-turth')
    plt.plot(x, xgb_pred_valid, 'r', label='xgb predict')
    plt.legend(loc='best')
    plt.title('xgb-yield')
    plt.xlabel('sample')
    plt.ylabel('Yield_lnRR')
    plt.show()

# %%
xgb_pred_test = xgb_model.predict(X_test)
print('The rmse of the XgbRegression is:',metrics.mean_squared_error(y_test, xgb_pred_test))
x = list(range(0, len(y_test)))
plt.figure(figsize=(40,10))
plt.plot(x, y_test, 'g', label='ground-turth')
plt.plot(x, xgb_pred_test, 'r', label='xgb predict')
plt.legend(loc='best')
plt.title('xgb-yield')
plt.xlabel('sample')
plt.ylabel('Yield_lnRR')
plt.show()

joblib.dump(xgb_model, f'{model_save_file}/xgb_model.pkl') 
result_model['XGBoost(k)'] = metrics.mean_squared_error(y_test,xgb_pred_test)

df_tmp = pd.DataFrame({'pred': xgb_pred_test, 'true': y_test})
df_tmp.to_csv(f'{model_save_file}/xgb_pred.csv', index=False)

# %% [markdown]
# #### LGB

# %%
import lightgbm as lgb

lgb_preds = np.zeros(X_train.shape[0])
for i, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
    print('************************************ {} ************************************'.format(str(i+1)))
    X_tr, y_tr, X_va, y_va = X_train.iloc[train_index], \
        y_train.iloc[train_index], X_train.iloc[valid_index], y_train.iloc[valid_index]

    params_lgb = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'learning_rate': 0.1,
                'n_estimators': 300,
                'metric': 'root_mean_squared_error',
                'min_child_weight': 1e-3,
                'min_child_samples': 10,
                'num_leaves': 31,
                'max_depth': -1,
                'seed': 2023,
                'verbose': -1,
    }
    
    lgb_model = lgb.LGBMRegressor(**params_lgb)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], eval_metric=['mae','rmse'])
    lgb_val_pred = lgb_model.predict(X_va)
    lgb_preds[valid_index] = lgb_val_pred

    print('The rmse of the LgbRegression is:',metrics.mean_squared_error(y_va,lgb_val_pred))
    x = list(range(0, len(y_va)))
    plt.figure(figsize=(40,10))
    plt.plot(x, y_va, 'g', label='ground-turth')
    plt.plot(x, lgb_val_pred, 'r', label='lgb predict')
    plt.legend(loc='best')
    plt.title('lgb-yield')
    plt.xlabel('sample')
    plt.ylabel('Yield_lnRR')
    plt.show()

# %%
lgb_pred_test = lgb_model.predict(X_testt)
print('The rmse of the test is:',metrics.mean_squared_error(y_testt,lgb_pred_test))
x = list(range(0, len(y_testt)))
plt.figure(figsize=(40,10))
plt.plot(x[:100], y_testt[:100], 'g', label='ground-turth')
plt.plot(x[:100], lgb_pred_test[:100], 'r', label='lgb predict')
plt.legend(loc='best')
plt.title('lgb-yield')
plt.xlabel('sample')
plt.ylabel('Yield_lnRR')
plt.show()

joblib.dump(lgb_model, f'{model_save_file}/lgb_model.pkl') 
result_model['Lightgbm(k)'] = metrics.mean_squared_error(y_testt,lgb_pred_test)

df_tmp = pd.DataFrame({'pred': lgb_pred_test, 'true': y_testt})
df_tmp.to_csv(f'{model_save_file}/lgb_pred.csv', index=False)

# %% [markdown]
# #### CTB

# %%
from catboost import CatBoostRegressor

params = {
    'learning_rate': 0.05,
    'loss_function': "RMSE",
    'eval_metric': "RMSE", # CrossEntropy
    'depth': 8,
    'min_data_in_leaf': 20,
    'random_seed': 42,
    'logging_level': 'Silent',
    'use_best_model': True,
    'one_hot_max_size': 5,   #类别数量多于此数将使用ordered target statistics编码方法,默认值为2。
    'boosting_type':"Ordered", #Ordered 或者Plain,数据量较少时建议使用Ordered,训练更慢但能够缓解梯度估计偏差。
    'max_ctr_complexity': 2, #特征组合的最大特征数量，设置为1取消特征组合，设置为2只做两个特征的组合,默认为4。
    'nan_mode': 'Min' 
}
iterations = 400
early_stopping_rounds = 200
ctb_model = CatBoostRegressor(iterations=iterations,
    early_stopping_rounds = early_stopping_rounds,
    **params)

for i, (train_index, valid_index) in enumerate(kf.split(X_train)):
    print('************************************ {} ************************************'.format(str(i+1)))
    x_tr, x_va = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_tr, y_va = y_train.iloc[train_index], y_train.iloc[valid_index]
    
    print(x_tr.shape, y_tr.shape)
    ctb_model.fit(x_tr, y_tr, eval_set=(x_va, y_va), verbose=0, early_stopping_rounds=1000)
    ctb_pre= ctb_model.predict(x_va)

    print('The rmse of the CatRegression is:',metrics.mean_squared_error(y_va,ctb_pre))
    x = list(range(0, len(y_va)))
    plt.figure(figsize=(40,10))
    plt.plot(x, y_va, 'g', label='ground-turth')
    plt.plot(x, ctb_pre, 'r', label='cat predict')
    plt.legend(loc='best')
    plt.title('cat-yield')
    plt.xlabel('sample')
    plt.ylabel('Yield_lnRR')
    plt.show()

# %%
cat_pred_test = ctb_model.predict(X_testt)
print('The rmse of the test is:',metrics.mean_squared_error(y_testt,cat_pred_test))
x = list(range(0, len(y_testt)))
plt.figure(figsize=(40,10))
plt.plot(x, y_testt, 'g', label='ground-turth')
plt.plot(x, cat_pred_test, 'r', label='cat predict')
plt.legend(loc='best')
plt.title('cat-yield')
plt.xlabel('sample')
plt.ylabel('Yield_lnRR')
plt.show()

joblib.dump(ctb_model, f'{model_save_file}/ctb_model.pkl') 
result_model['Catboost(k)'] = metrics.mean_squared_error(y_test,cat_pred_test)

df_tmp = pd.DataFrame({'pred': cat_pred_test, 'true': y_test})
df_tmp.to_csv(f'{model_save_file}/cat_pred.csv', index=False)

# %% [markdown]
# ### 使用其他机器学习方法预测

# %% [markdown]
# #### 岭回归

# %%
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

ri_model = Ridge(alpha=0.5,fit_intercept=True)

lr_preds = np.zeros(X_train.shape[0])
for train_index, valid_index in kf.split(X_train, y_train):
    x_tr, x_va = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_tr, y_va = y_train.iloc[train_index], y_train.iloc[valid_index]

    print(x_tr.shape, y_tr.shape)
    ri_model.fit(x_tr, y_tr)
    lr_pred_valid = ri_model.predict(x_va)
    lr_preds[valid_index] = lr_pred_valid

    print('The rmse of the LR is:',metrics.mean_squared_error(y_va,lr_pred_valid))
    x = list(range(0, len(y_va)))
    plt.figure(figsize=(40,10))
    plt.plot(x, y_va, 'g', label='ground-turth')
    plt.plot(x, lr_pred_valid, 'r', label='lr predict')
    plt.legend(loc='best')
    plt.title('lr-yield')
    plt.xlabel('sample')
    plt.ylabel('Yield_lnRR')
    plt.show()

# %%
ri_pred_test = ri_model.predict(X_test)
print('The rmse of the test is:',metrics.mean_squared_error(y_test,ri_pred_test))
x = list(range(0, len(y_test)))
plt.figure(figsize=(40,10))
plt.plot(x, y_test, 'g', label='ground-turth')
plt.plot(x, ri_pred_test, 'r', label='lr predict')
plt.legend(loc='best')
plt.title('lr-yield')
plt.xlabel('sample')
plt.ylabel('Yield_lnRR')
plt.show()

joblib.dump(ri_model, f'{model_save_file}/ri_model.pkl') 
result_model['RidgeRegression(k)'] = metrics.mean_squared_error(y_test,ri_pred_test)

# %% [markdown]
# #### 线性回归

# %%
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

lr_model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)

lr_preds = np.zeros(X_train.shape[0])
for train_index, valid_index in kf.split(X_train, y_train):
    x_tr, x_va = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_tr, y_va = y_train.iloc[train_index], y_train.iloc[valid_index]

    print(x_tr.shape, y_tr.shape)
    lr_model.fit(x_tr, y_tr)
    lr_pred_valid = lr_model.predict(x_va)
    lr_preds[valid_index] = lr_pred_valid

    # print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_valid,xgb_pred_valid))
    print('The rmse of the LR is:',metrics.mean_squared_error(y_va,lr_pred_valid))
    x = list(range(0, len(y_va)))
    plt.figure(figsize=(40,10))
    plt.plot(x, y_va, 'g', label='ground-turth')
    plt.plot(x, lr_pred_valid, 'r', label='lr predict')
    plt.legend(loc='best')
    plt.title('lr-yield')
    plt.xlabel('sample')
    plt.ylabel('Yield_lnRR')
    plt.show()

# %%
lr_pred_test = lr_model.predict(X_test)
print('The rmse of the test is:',metrics.mean_squared_error(y_test,lr_pred_test))
x = list(range(0, len(y_test)))
plt.figure(figsize=(40,10))
plt.plot(x, y_test, 'g', label='ground-turth')
plt.plot(x, lr_pred_test, 'r', label='lr predict')
plt.legend(loc='best')
plt.title('lr-yield')
plt.xlabel('sample')
plt.ylabel('Yield_lnRR')
plt.show()

joblib.dump(lr_model, f'{model_save_file}/lr_model.pkl')
result_model['LinearRegression(k)'] = metrics.mean_squared_error(y_test,lr_pred_test)

df_tmp = pd.DataFrame({'pred': lr_pred_test, 'true': y_test})
df_tmp.to_csv(f'{model_save_file}/lr_pred.csv', index=False)

# %% [markdown]
# #### 使用MLP进行预测

# %%
from sklearn.neural_network import MLPRegressor

mlp_model = MLPRegressor(solver='lbfgs',alpha=1e-5, hidden_layer_sizes=(40,40), max_iter=500, random_state=2023)

mlp_preds = np.zeros(X_train.shape[0])
for train_index, valid_index in kf.split(X_train, y_train):
    x_tr, x_va = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_tr, y_va = y_train.iloc[train_index], y_train.iloc[valid_index]

    print(x_tr.shape, y_tr.shape)
    mlp_model.fit(x_tr, y_tr)
    mlp_pred_valid = mlp_model.predict(x_va)
    mlp_preds[valid_index] = mlp_pred_valid

    # print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_valid,xgb_pred_valid))
    print('The rmse of the MLP is:',metrics.mean_squared_error(y_va,mlp_pred_valid))
    x = list(range(0, len(y_va)))
    plt.figure(figsize=(40,10))
    plt.plot(x, y_va, 'g', label='ground-turth')
    plt.plot(x, mlp_pred_valid, 'r', label='mlp predict')
    plt.legend(loc='best')
    plt.title('mlp-yield')
    plt.xlabel('sample')
    plt.ylabel('Yield_lnRR')
    plt.show()

# %%
mlp_pred_test = mlp_model.predict(X_test)
print('The rmse of the test is:',metrics.mean_squared_error(y_test,mlp_pred_test))
x = list(range(0, len(y_test)))
plt.figure(figsize=(40,10))
plt.plot(x, y_test, 'g', label='ground-turth')
plt.plot(x, mlp_pred_test, 'r', label='mlp predict')
plt.legend(loc='best')
plt.title('mlp-yield')
plt.xlabel('sample')
plt.ylabel('Yield_lnRR')
plt.show()

joblib.dump(mlp_model, f'{model_save_file}/mlp_model.pkl')
result_model['MLP(k)'] = metrics.mean_squared_error(y_test,mlp_pred_test)

# %% [markdown]
# #### SVR进行预测

# %%
from sklearn.svm import SVR

svr_model = SVR(kernel ='rbf',
                degree = 3,
                coef0 = 0.0,
                tol = 0.001,
                C = 1.0,
                epsilon = 0.1,
                shrinking = True,
                cache_size = 200,
                verbose = False,
                max_iter = -1)

svr_preds = np.zeros(X_train.shape[0])
for train_index, valid_index in kf.split(X_train, y_train):
    x_tr, x_va = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_tr, y_va = y_train.iloc[train_index], y_train.iloc[valid_index]

    print(x_tr.shape, y_tr.shape)
    svr_model.fit(x_tr, y_tr)
    svr_pred_valid = svr_model.predict(x_va)
    svr_preds[valid_index] = svr_pred_valid

    # print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_valid,xgb_pred_valid))
    print('The rmse of the LR is:',metrics.mean_squared_error(y_va,svr_pred_valid))
    x = list(range(0, len(y_va)))
    plt.figure(figsize=(40,10))
    plt.plot(x, y_va, 'g', label='ground-turth')
    plt.plot(x, svr_pred_valid, 'r', label='svr predict')
    plt.legend(loc='best')
    plt.title('svr-yield')
    plt.xlabel('sample')
    plt.ylabel('Yield_lnRR')
    plt.show()

# %%
svr_pred_test = svr_model.predict(X_test)
print('The rmse of the test is:',metrics.mean_squared_error(y_test,svr_pred_test))
x = list(range(0, len(y_test)))
plt.figure(figsize=(40,10))
plt.plot(x, y_test, 'g', label='ground-turth')
plt.plot(x, svr_pred_test, 'r', label='mlp predict')
plt.legend(loc='best')
plt.title('mlp-yield')
plt.xlabel('sample')
plt.ylabel('Yield_lnRR')
plt.show()

joblib.dump(svr_model, f'{model_save_file}/svr_model.pkl')
result_model['SVR(k)'] = metrics.mean_squared_error(y_test,svr_pred_test)

# %% [markdown]
# #### 使用GAUSS Rgression进行预测

# %%
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# 创建高斯过程回归模型对象
gp_model = GaussianProcessRegressor(kernel=RBF())

gp_preds = np.zeros(X_train.shape[0])
for train_index, valid_index in kf.split(X_train, y_train):
    x_tr, x_va = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_tr, y_va = y_train.iloc[train_index], y_train.iloc[valid_index]

    print(x_tr.shape, y_tr.shape)
    gp_model.fit(x_tr, y_tr)
    gp_pred_valid = gp_model.predict(x_va)
    gp_preds[valid_index] = gp_pred_valid

    print('The rmse of the LR is:',metrics.mean_squared_error(y_va,gp_pred_valid))
    x = list(range(0, len(y_va)))
    plt.figure(figsize=(40,10))
    plt.plot(x, y_va, 'g', label='ground-turth')
    plt.plot(x, gp_pred_valid, 'r', label='gp predict')
    plt.legend(loc='best')
    plt.title('gp-yield')
    plt.xlabel('sample')
    plt.ylabel('Yield_lnRR')
    plt.show()

# %%
gp_pred_test = gp_model.predict(X_test)
print('The rmse of the test is:',metrics.mean_squared_error(y_test,gp_pred_test))
x = list(range(0, len(y_test)))
plt.figure(figsize=(40,10))
plt.plot(x, y_test, 'g', label='ground-turth')
plt.plot(x, gp_pred_test, 'r', label='mlp predict')
plt.legend(loc='best')
plt.title('mlp-yield')
plt.xlabel('sample')
plt.ylabel('Yield_lnRR')
plt.show()

joblib.dump(gp_model, f'{model_save_file}/gp_model.pkl')
result_model['GR(k)'] = metrics.mean_squared_error(y_test,gp_pred_test)

# %% [markdown]
# ### 不同模型的效果比对分析（RMSE指标）

# %%
import json

# 将字典保存为 JSON 文件
with open(model_performance_json, 'w') as json_file:
    json.dump(result_model, json_file)

# %%
import plotly.express as px
import plotly.io as pio

with open(model_performance_json, 'r') as json_file:
    result_model = json.load(json_file)
    
result_model = dict(sorted(result_model.items(), key=lambda item: item[1]))
categories = list(result_model.keys())
values = list(result_model.values())

color_mapping = {}
for k, v in result_model.items():
    if v < 0.4:
        color_mapping[k] = "Tree-Base"
    else:
        color_mapping[k] = "Other"
    
# 创建柱状图
fig = px.bar(x=categories, y=values, title='Models Performance in ln(EF)', color=color_mapping)
fig.update_layout(template="seaborn")
# 显示图表
fig.show()
# 保存柱状图为 HTML 文件
pio.write_html(fig, file=model_performance_path)  

# %% [markdown]
# ### 挑选合适的模型进行进一步的变量分析(Tree-Base)

# %%
import shap
shap.initjs()

# %% [markdown]
# #### 变量重要性分析

# %% [markdown]
# ##### RF模型分析--变量重要性分析

# %%
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

temp = dict(layout=go.Layout(font=dict(family="Franklin Gothic", size=12), width=800))
colors=px.colors.qualitative.Plotly

feat_importance=pd.DataFrame()
feat_importance["Importance"]=rf_model.feature_importances_
feat_importance.set_index(X_train.columns, inplace=True)

feat_importance = feat_importance.sort_values(by='Importance',ascending=True)
pal=sns.color_palette("plasma_r", 70).as_hex()[2:]

fig=go.Figure()
for i in range(len(feat_importance.index)):
    fig.add_shape(dict(type="line", y0=i, y1=i, x0=0, x1=feat_importance['Importance'][i], 
                       line_color=pal[::-1][i],opacity=0.7,line_width=4))
fig.add_trace(go.Scatter(x=feat_importance['Importance'], y=feat_importance.index, mode='markers', 
                         marker_color=pal[::-1], marker_size=8,
                         hovertemplate='%{y} Importance = %{x:.5f}<extra></extra>'))
fig.update_layout(template=temp,title='Overall Feature Importance', 
                  xaxis=dict(title='Average Importance',zeroline=False),
                  yaxis_showgrid=False, margin=dict(l=120,t=80),
                  height=700, width=800)
fig.show()

# %% [markdown]
# ##### Xgboost

# %%
feat_importance=pd.DataFrame()
feat_importance["Importance"]=xgb_model.feature_importances_
feat_importance.set_index(X_train.columns, inplace=True)

feat_importance = feat_importance.sort_values(by='Importance',ascending=True)
pal=sns.color_palette("plasma_r", 70).as_hex()[2:]

fig=go.Figure()
for i in range(len(feat_importance.index)):
    fig.add_shape(dict(type="line", y0=i, y1=i, x0=0, x1=feat_importance['Importance'][i], 
                       line_color=pal[::-1][i],opacity=0.7,line_width=4))
fig.add_trace(go.Scatter(x=feat_importance['Importance'], y=feat_importance.index, mode='markers', 
                         marker_color=pal[::-1], marker_size=8,
                         hovertemplate='%{y} Importance = %{x:.5f}<extra></extra>'))
fig.update_layout(template=temp,title='Overall Feature Importance', 
                  xaxis=dict(title='Average Importance',zeroline=False),
                  yaxis_showgrid=False, margin=dict(l=120,t=80),
                  height=700, width=800)
fig.show()

# %% [markdown]
# ##### Catboost

# %%
feat_importance=pd.DataFrame()
feat_importance["Importance"]=ctb_model.feature_importances_
feat_importance.set_index(X_train.columns, inplace=True)

feat_importance = feat_importance.sort_values(by='Importance',ascending=True)
pal=sns.color_palette("plasma_r", 70).as_hex()[2:]

fig=go.Figure()
for i in range(len(feat_importance.index)):
    fig.add_shape(dict(type="line", y0=i, y1=i, x0=0, x1=feat_importance['Importance'][i], 
                       line_color=pal[::-1][i],opacity=0.7,line_width=4))
fig.add_trace(go.Scatter(x=feat_importance['Importance'], y=feat_importance.index, mode='markers', 
                         marker_color=pal[::-1], marker_size=8,
                         hovertemplate='%{y} Importance = %{x:.5f}<extra></extra>'))
fig.update_layout(template=temp,title='Overall Feature Importance', 
                  xaxis=dict(title='Average Importance',zeroline=False),
                  yaxis_showgrid=False, margin=dict(l=120,t=80),
                  height=700, width=800)
fig.show()

# %% [markdown]
# ##### Lightbgm

# %%
feat_importance=pd.DataFrame()
feat_importance["Importance"]=lgb_model.feature_importances_
feat_importance.set_index(X_train.columns, inplace=True)

feat_importance = feat_importance.sort_values(by='Importance',ascending=True)
pal=sns.color_palette("plasma_r", 70).as_hex()[2:]

fig=go.Figure()
for i in range(len(feat_importance.index)):
    fig.add_shape(dict(type="line", y0=i, y1=i, x0=0, x1=feat_importance['Importance'][i], 
                       line_color=pal[::-1][i],opacity=0.7,line_width=4))
fig.add_trace(go.Scatter(x=feat_importance['Importance'], y=feat_importance.index, mode='markers', 
                         marker_color=pal[::-1], marker_size=8,
                         hovertemplate='%{y} Importance = %{x:.5f}<extra></extra>'))
fig.update_layout(template=temp,title='Overall Feature Importance', 
                  xaxis=dict(title='Average Importance',zeroline=False),
                  yaxis_showgrid=False, margin=dict(l=120,t=80),
                  height=700, width=800)
fig.show()

# %% [markdown]
# SHAP是由Shapley value启发的可加性解释模型。对于每个预测样本，模型都产生一个预测值，SHAP value就是该样本中每个特征所分配到的数值。 
# 很明显可以看出，与上一节中feature importance相比，SHAP value最大的优势是SHAP能对于反映出每一个样本中的特征的影响力，而且还表现出影响的正负性。

# %%
# 获取feature importance
plt.figure(figsize=(15, 5))

feat_importance=pd.DataFrame()
feat_importance["Importance"]=lgb_model.feature_importances_
feat_importance.set_index(X_train.columns, inplace=True)

feat_importance = feat_importance.sort_values(by='Importance', ascending=False)

plt.bar(range(len(X_train.columns)), feat_importance['Importance'])
plt.xticks(range(len(X_train.columns)), feat_importance.index, rotation=90, fontsize=14)
plt.title('Feature importance', fontsize=14)
plt.show()

# %%
lgb_explainer = shap.TreeExplainer(lgb_model)
lgb_shap_values = lgb_explainer.shap_values(X_train)
print(lgb_shap_values.shape)

# %%
shap.force_plot(lgb_explainer.expected_value, lgb_shap_values[0,:], X_train.iloc[0,:])

# %% [markdown]
# 对第一个实例的特征贡献图也可用 waterfall 方式展示

# %%
shap.plots._waterfall.waterfall_legacy(
    lgb_explainer.expected_value, 
    lgb_shap_values[0], 
    feature_names=X_train.columns)

# %% [markdown]
# 上图的解释显示了每个有助于将模型输出从基值（我们传递的训练数据集上的平均模型输出）贡献到模型输出值的特征。将预测推高的特征以红色显示，将预测推低的特征以蓝色显示。
# 
# 如果我们采取许多实例来聚合显示解释，如下图所示，将它们旋转 90 度，然后将它们水平堆叠，我们可以看到整个数据集的解释（在 Notebook 中，此图是交互式的）

# %%
# visualize the training set predictions
shap.force_plot(lgb_explainer.expected_value, lgb_shap_values, X_train)

# %% [markdown]
# 下图中每一行代表一个特征，横坐标为SHAP值。一个点代表一个样本，颜色越红说明特征本身数值越大，颜色越蓝说明特征本身数值越小。

# %%
shap.summary_plot(lgb_shap_values, X_train)

# %% [markdown]
# shap value值排序的特征重要性

# %%
shap.summary_plot(lgb_shap_values, X_train, plot_type="bar")

# %% [markdown]
# SHAP也提供了部分依赖图的功能，与传统的部分依赖图不同的是，这里纵坐标不是目标变量y的数值而是SHAP值。可以观察各个特征的分布与目标shap值的关系。

# %%
fig, axes = plt.subplots(len(input_cols)//3+1, 3, figsize=(20,30))
for i, col in enumerate(input_cols):
    shap.dependence_plot(col, lgb_shap_values, X_train, interaction_index=None, show=False, ax=axes[i//3,i%3])
    

# %% [markdown]
# - 对多个变量的交互进行分析
# 
# 我们也可以多个变量的交互作用进行分析。一种方式是采用summary_plot描绘出散点图，如下：

# %%
shap_interaction_values = shap.TreeExplainer(lgb_model).shap_interaction_values(X_train)
plt.figure(figsize=(12,12))
shap.summary_plot(shap_interaction_values, X_train, max_display=6)
plt.show()

# %% [markdown]
# 我们也可以用dependence_plot描绘两个变量交互下变量对目标值的影响。

# %%
shap.dependence_plot(input_cols[0], lgb_shap_values, X_train, interaction_index=input_cols[1], show=False)

# %% [markdown]
# 也能可视化每种特征对于整体预测值的影响。

# %%
shap.decision_plot(lgb_explainer.expected_value, lgb_shap_values, X_train, ignore_warnings=True)

# %% [markdown]
# ##### Catboost

# %%
feat_importance=pd.DataFrame()
feat_importance["Importance"]=ctb_model.feature_importances_
feat_importance.set_index(X_train.columns, inplace=True)

feat_importance = feat_importance.sort_values(by='Importance',ascending=True)
pal=sns.color_palette("plasma_r", 70).as_hex()[2:]

fig=go.Figure()
for i in range(len(feat_importance.index)):
    fig.add_shape(dict(type="line", y0=i, y1=i, x0=0, x1=feat_importance['Importance'][i], 
                       line_color=pal[::-1][i],opacity=0.7,line_width=4))
fig.add_trace(go.Scatter(x=feat_importance['Importance'], y=feat_importance.index, mode='markers', 
                         marker_color=pal[::-1], marker_size=8,
                         hovertemplate='%{y} Importance = %{x:.5f}<extra></extra>'))
fig.update_layout(template=temp,title='Overall Feature Importance', 
                  xaxis=dict(title='Average Importance',zeroline=False),
                  yaxis_showgrid=False, margin=dict(l=120,t=80),
                  height=700, width=800)
fig.show()


