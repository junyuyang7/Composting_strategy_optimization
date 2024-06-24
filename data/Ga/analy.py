import pandas as pd

# 读取CSV文件
data_gi = pd.read_csv('data\\Ga\\GI__04\\data_for_Final GI (%).csv')
data_ch4_loss = pd.read_csv('data\\Ga\\__08\\data_for_CH4-C loss (%).csv')

# 打印数据的前几行以了解其结构
print("Data for Final GI (%):")
print(data_gi.head())
print("\nData for CH4-C loss (%):")
print(data_ch4_loss.head())

# 打印数据的基本信息以了解其特征
print("\nData for Final GI (%) Info:")
print(data_gi.info())
print("\nData for CH4-C loss (%) Info:")
print(data_ch4_loss.info())

# 比较两组数据的列名
columns_gi = set(data_gi.columns)
columns_ch4_loss = set(data_ch4_loss.columns)

# 打印两个数据集的列名
print("\nColumns in Data for Final GI (%):")
print(columns_gi)
print("\nColumns in Data for CH4-C loss (%):")
print(columns_ch4_loss)

# 找出共同的列和不同的列
common_columns = columns_gi.intersection(columns_ch4_loss)
diff_columns_gi = columns_gi.difference(columns_ch4_loss)
diff_columns_ch4_loss = columns_ch4_loss.difference(columns_gi)

print("\nCommon columns:")
print(common_columns)
print("\nColumns only in Data for Final GI (%):")
print(diff_columns_gi)
print("\nColumns only in Data for CH4-C loss (%):")
print(diff_columns_ch4_loss)

# 比较两组数据的描述性统计信息
print("\nDescriptive statistics for Data for Final GI (%):")
print(data_gi.describe())
print("\nDescriptive statistics for Data for CH4-C loss (%):")
print(data_ch4_loss.describe())

# 找出全部是整型的列并打印
int_columns_gi = {col for col in data_gi.columns if pd.api.types.is_integer_dtype(data_gi[col])}
int_columns_ch4_loss = {col for col in data_ch4_loss.columns if pd.api.types.is_integer_dtype(data_ch4_loss[col])}

# 合并整型列名并打印
all_int_columns = list(int_columns_gi.union(int_columns_ch4_loss))

print("\nInteger columns in Data for Final GI (%):")
print(int_columns_gi)
print("\nInteger columns in Data for CH4-C loss (%):")
print(int_columns_ch4_loss)



print("\nAll integer columns in both datasets:")
print(all_int_columns)