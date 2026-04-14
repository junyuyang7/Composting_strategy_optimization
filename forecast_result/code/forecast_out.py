import pickle
import joblib
import pandas as pd
import os

targets = ["Final GI (%)", 'NH3-N loss (%)', 'N2O-N loss (%)', 'CH4-C loss (%)', 'CO2-C loss (%)']


model_file_dict = {
    "Final GI (%)":"output/raw_data_selected_0625_useall_True/__18/model_Final GI (%)/rf/rf_model.pkl",
    "NH3-N loss (%)":"output/raw_data_selected_0625_useall_True/__16/model_NH3-N loss (%)/lgb/lgb_model.pkl",
    "N2O-N loss (%)":"output/raw_data_selected_0625_useall_True/__16/model_N2O-N loss (%)/xgb/xgb_model.pkl",
    "CH4-C loss (%)":"output/raw_data_selected_0625_useall_True/__16/model_CH4-C loss (%)/cat/cat_model.pkl",
    "CO2-C loss (%)":"output/raw_data_selected_0625_useall_True/__16/model_CO2-C loss (%)/cat/cat_model.pkl",
}

data_input_cols = {
    "Final GI (%)":"forecast_result/data/mean_data/data_for_Final GI (%).csv",
    "NH3-N loss (%)":"forecast_result/data/mean_data/data_for_NH3-N loss (%).csv",
    "N2O-N loss (%)":"forecast_result/data/mean_data/data_for_N2O-N loss (%).csv",
    "CH4-C loss (%)":"forecast_result/data/mean_data/data_for_CH4-C loss (%).csv",
    "CO2-C loss (%)":"forecast_result/data/mean_data/data_for_CO2-C loss (%).csv",   
}
# 初始化一个空列表来存储DataFrame
csv_files = "forecast_result/data/fore_data/converted_fore_data.csv"

df = pd.read_csv(csv_files)
model_features = []
# 加载模型
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
    return model

# 预测函数
def predict(model, input_data):
    # 这里需要根据你的模型来调整预测逻辑
    # 例如，如果是一个scikit-learn模型，你可以直接调用model.predict(input_data)
    prediction = model.predict(input_data)
    return prediction

# 加载模型并预测结果的函数
def load_and_predict(target, model_path, df):
    # 加载模型
    model = load_model(model_path)

    model_features = pd.read_csv(data_input_cols[target]).columns.tolist()
    model_features = [column for column in model_features if column != target]
    print('>>>=======================================<<<')
    print(model_features)
    print(len(model_features))
    print('>>>=======================================<<<')
    # 确保 input_data 仅包含模型训练时使用的特征列
    input_data = df[model_features]
    print('>>>=======================================<<<')
    print(input_data.head())
    print(len(input_data))
    print('>>>=======================================<<<')
    # 进行预测
    predictions = predict(model, input_data)

    # 将预测结果添加到 df 中，替换原有的目标列
    df[target] = predictions
    
    return df

# 预测并更新DataFrame
for target in targets:
    model_path = model_file_dict[target]
    df = load_and_predict(target, model_path, df)
out_file = 'forecast_result/result/result.csv'
# 将更新后的DataFrame保存回CSV文件
df.to_csv(out_file, index=False)
print(f"更新后的预测结果已保存到 {csv_files}")