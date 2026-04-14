import pandas as pd

# 读取Excel文件
file_path = "data/data_0625/堆肥数据库-机器学习v5.xlsx"
sheet_name = "18"
df_excel = pd.read_excel(file_path, sheet_name=sheet_name)

# 读取CSV文件
csv_file_path = "forecast_result/data/fore_data/ref_mean_tbl.csv"
df_csv = pd.read_csv(csv_file_path)

# 指定需要计算平均值的列
columns_to_average = [
    "Initial Moisture Content (%)",
    "Initial pH",
    "Initial TN (%)",
    "Initial TC (%)",
    "Initial C/N (%)",
    "Compost volume (m3)",
    "Initial GI (%)",
    'Initial density (kg/L)'
]

# 根据Material_Main对数据进行分组，并计算每组的平均值
grouped_avg = df_excel.groupby('Material_Main')[columns_to_average].mean().reset_index()
grouped_avg.to_csv(csv_file_path, index=False)

# # 将计算得到的平均值与CSV文件中的Material_Main进行匹配，并填充到对应的列上
# for material_main in grouped_avg['Material_Main'].unique():
#     mask = df_csv['Material_Main'] == material_main
#     for column in columns_to_average:
#         avg_value = grouped_avg[grouped_avg['Material_Main'] == material_main][column].iloc[0]
#         df_csv.loc[mask, column] = avg_value

# # 将更新后的DataFrame保存回CSV文件
# df_csv.to_csv(csv_file_path, index=False)
# print(f"Updated data has been saved to {csv_file_path}")