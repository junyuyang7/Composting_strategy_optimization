import os
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer

class Data_Pro:
    # 利用方差检验法检查异常值 三倍方差法
    def find_outlines_by_3segama(df, fea, times = 3):
        data_std = np.std(df[fea])
        data_mean = np.mean(df[fea])
        outlines_cut_off = data_std * times
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
    
    def fill_missing(df, numerical_fea, categorical_fea, integer_columns, fillcontent=0, n_neighbors=10):
        """
        填充数据框中的缺失值，数值特征使用KNN填充，非数值特征使用众数填充，整型特征使用中位数填充。

        参数：
        - df: 数据框
        - numerical_fea: 数值特征列列表
        - categorical_fea: 非数值特征列列表
        - integer_columns: 整型特征列列表
        - fillcontent: 填充数值特征的默认值（未使用）
        - n_neighbors: KNN填充器的邻居数量

        返回：
        - 填充后的数据框
    """
    
        # 创建一个副本来保存所有填充后的数据
        df_filled_all = df.copy()

        # 处理数值特征列
        if numerical_fea:
            df_numerical = df[numerical_fea]
            # 输出原始缺失值信息
            print("原始数值特征缺失值：")
            print(df_numerical.isnull().sum())
            
            # 初始化归一化器
            scaler = MinMaxScaler()
            
            # 对数据进行归一化
            df_numerical_scaled = scaler.fit_transform(df_numerical)
            
            # 初始化KNN填充器
            imputer = KNNImputer(n_neighbors=n_neighbors)
            
            # 对数据进行填充
            df_filled_scaled = imputer.fit_transform(df_numerical_scaled)
            
            # 反归一化填充后的数据
            df_filled = scaler.inverse_transform(df_filled_scaled)
            
            # 将填充后的数据重新放回DataFrame
            df_filled = pd.DataFrame(df_filled, columns=numerical_fea, index=df_numerical.index)
            
            # 输出填充后的缺失值信息
            print("填充后数值特征缺失值：")
            print(df_filled.isnull().sum())
            
            # 将填充后的数值特征列合并回原数据框
            for col in numerical_fea:
                print(f"更新数值特征列：{col}")
                df_filled_all[col] = df_filled[col]

        # 处理非数值特征列
        categorical_fea_with_no_integer = [col for col in categorical_fea if col not in integer_columns]
        if categorical_fea_with_no_integer:
            df_categorical = df[categorical_fea_with_no_integer]
            
            # 初始化众数填充器
            imputer = SimpleImputer(strategy='most_frequent')
            
            # 对数据进行填充
            df_filled = imputer.fit_transform(df_categorical)
            
            # 将填充后的数据重新放回DataFrame
            df_filled = pd.DataFrame(df_filled, columns=categorical_fea_with_no_integer, index=df_categorical.index)
            
            # 输出填充后的缺失值信息
            print("填充后非数值特征缺失值：")
            print(df_filled.isnull().sum())
            
            # 将填充后的非数值特征列合并回原数据框
            for col in categorical_fea_with_no_integer:
                print(f"更新非数值特征列：{col}")
                df_filled_all[col] = df_filled[col]

        # 处理整型特征列
        categorical_fea_with_integer = [col for col in categorical_fea if col in integer_columns]
        if categorical_fea_with_integer:
            df_integer = df[categorical_fea_with_integer]
            
            # 初始化中位数填充器
            imputer = SimpleImputer(strategy='median')
            
            # 对数据进行填充
            df_filled = imputer.fit_transform(df_integer)
            
            # 将填充后的数据重新放回DataFrame
            df_filled = pd.DataFrame(df_filled, columns=categorical_fea_with_integer, index=df_integer.index)
            
            # 输出填充后的缺失值信息
            print("填充后整型特征缺失值：")
            print(df_filled.isnull().sum())
            
            # 将填充后的整型特征列合并回原数据框
            for col in categorical_fea_with_integer:
                print(f"更新整型特征列：{col}")
                df_filled_all[col] = df_filled[col]
        
        return df_filled_all
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
    
    # 区分高低类别特征
    def high_or_low_category(df, columns, target):
        high_cat_features = []
        low_cat_features = []

        # 只考虑目标列非空行
        filtered_df = df[df[target].notna()]

        for col in columns:
            if df[col].dtype == 'object':
                unique_values = filtered_df[col].dropna().unique()  # 去除NaN后获取唯一值
                unique_count = len(unique_values)  # 计算唯一值数量
                print(f"特征 '{col}' 有 {unique_count} 个类别: {unique_values}")  # 打印特征名、类别数量和特征值
                
                # 类别特征大于等于20认为是高基数类别特征
                if unique_count >= 100:
                    high_cat_features.append(col)
                else:
                    low_cat_features.append(col)
        
        return high_cat_features, low_cat_features
    # 序列编码
    def encode_categories(df, columns, save_path, target):
        dic_lst = []  # 返回映射关系的字典
        for col in columns:
            df_nonan = df.dropna(subset=[col])
            label_encoder = LabelEncoder()
            # 对类别数据进行编码
            encoded_column = label_encoder.fit_transform(df_nonan[col])
            df_nonan[col] = encoded_column
            # 返回类别到数字的映射字典
            category_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            dic_lst.append(category_mapping)
            # 保存编码表
            file_path = os.path.join(save_path, target[0], f'{col}_ordinal_encoding.csv')
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            encode_table = pd.DataFrame(list(category_mapping.items()), columns=[col, f'{col}_encoded'])
            encode_table.to_csv(file_path, index=False)
            # 更新原始数据框
            df[col].update(df_nonan[col])
        return df, dic_lst

    # 平均数编码
    def mean_encoding(df, cat_feature, target_feature, save_path):
        for cat in cat_feature:
            for trg in target_feature:
                try:
                    # 去除包含 NaN 的行
                    df_nonan = df.dropna(subset=[cat, trg])
                    
                    # 计算每个类别的均值
                    mean_encoded_values = df_nonan.groupby(cat)[trg].mean()
                    
                    # 将均值映射到原数据框，保留原列名
                    df[cat] = df[cat].map(mean_encoded_values)
                    
                    # 保存编码表
                    encode_table = mean_encoded_values.reset_index()
                    encode_table.columns = [cat, f'{cat}_mean_{trg}']

                    file_path = os.path.join(save_path, target_feature[0],f'{cat}_mean_encoding.csv')
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    encode_table.to_csv(file_path, index=False)
                except Exception as e:
                    print('mean_encoding出现错误：', e)
                    print('cat:', cat, 'trg:', trg)

        # 返回包含所有原始特征和目标特征的数据框
        return df
    
    def remove_outliers_knn(df_raw, feas, filled=False, n_neighbors=50):
        """
        使用LOF算法去除或填充异常值。

        参数：
        - df_raw: 原始数据帧
        - feas: 需要检测和处理异常值的特征列表
        - filled: 是否填充离群点的特征值（默认为False）
        - n_neighbors: 用于LOF算法的邻居数量（默认为50）

        返回：
        - 处理后的数据帧
        - 离群点的索引列表
        """
        # 提取需要检测异常值的特征数据
        df = df_raw[feas]

        # 确保n_neighbors的值在合理范围内
        n_neighbors = min(n_neighbors, len(df) // 2)  # 避免n_neighbors过大或过小
        if n_neighbors < 5:
            print("数据点太少，无法有效进行异常检测。跳过去除异常值。")
            return df_raw, []

        # 初始化LOF模型
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)

        # 拟合模型并预测离群点
        outliers = lof.fit_predict(df)

        # 获取离群点的索引
        outlier_indices = np.where(outliers == -1)[0]
        
        # 打印离群点信息
        print(f"检测到的离群点索引: {outlier_indices}")
        print(f"检测到的离群点数据:\n{df_raw.iloc[outlier_indices]}")

        # 如果需要填充离群点的特征值
        if filled:
            # 提取离群点数据
            df_fill = df_raw[outliers == -1].copy()
            
            # 对每个特征进行处理，用非离群点的特征均值填充离群点的特征值
            for fea in feas:
                non_outlier_mean = df_raw.loc[outliers == 1, fea].mean()
                df_fill[fea] = non_outlier_mean
            
            # 更新原始数据集中的离群点数据
            df_raw.update(df_fill)
        else:
            # 如果不填充，则移除离群点
            df_raw = df_raw[outliers == 1]
        
        return df_raw, outlier_indices
    
    
    def change(data_all, numeric_factors, target):
        for col in data_all.columns.to_list():
            data_all[col] = data_all[col].astype(str)
        for col in numeric_factors:
            data_all[col] = pd.to_numeric(data_all[col], errors='coerce')
        for col in target:
            data_all[col] = pd.to_numeric(data_all[col], errors='coerce')
            
    def read_and_split_columns(input_file, encodings, sheet_name=None):
        data_all = None
        
        # 尝试用不同的编码读取文件
        for encoding in encodings:
            try:
                if input_file.endswith('.csv'):
                    data_all = pd.read_csv(input_file, encoding=encoding)
                else:
                    data_all = pd.read_excel(input_file, sheet_name=sheet_name)
                break  # 如果成功读取数据，则跳出循环
            except Exception as e:
                print(f"An error occurred with encoding {encoding}: {e}")
        nan_list = ['-',"None"]
        # 替换 data_all 中既包含数值又包含 nan_list 值的列中的 nan_list 值为 0

        # 替换 data_all 中不是全字符串列中的 nan_list 值为 0
        for col in data_all.columns:
            # 检查该列是否全是字符串
            if not all(isinstance(x, str) for x in data_all[col].dropna().unique()):
                # 将 nan_list 中的值替换为 0
                data_all[col].replace(nan_list, 0, inplace=True)

        # 输出替换后的数据框架
        print(data_all.head())

    
        # 初始化数值列和非数值列的列表
        integer_columns = []
        numerical_columns = []
        non_numerical_columns = []

        for column in data_all.columns:
            try:
                # 过滤非空值
                non_empty_values = data_all[column].dropna()
                # 尝试将非空值转换为数值类型
                converted_values = pd.to_numeric(non_empty_values, errors='coerce')
                # 找出无法转换为数值的原始值
                non_numeric_values = pd.Series(non_empty_values)[converted_values.isna()]

                print("无法转换为数值的值:")
                print(non_numeric_values)
                if converted_values.notna().all():
                    print(column)
                    # print((converted_values % 1)[converted_values % 1 != 0])
                    if (converted_values % 1 == 0).all():
                        print(f"---------------------converted_values % 1 == 0:{column}-----------------------")
                        integer_columns.append(column)  # 整数类型视为非数值类型
                        non_numerical_columns.append(column)  # 整数类型视为非数值类型
                    else:
                        print(f"---------------------numerical_columns:{column}-----------------------")
                        numerical_columns.append(column)
                else:
                    print(f"---------------------non_numerical_columns:{column}-----------------------")
                    non_numerical_columns.append(column)
                    
            except ValueError:
                print(f"---------------------ValueError:{column}-----------------------")
                non_numerical_columns.append(column)

        print('Numerical columns:', numerical_columns)
        print('Non-numerical (including integer) columns:', non_numerical_columns)
        print('integer_columns:', integer_columns)
        return data_all, numerical_columns, non_numerical_columns, integer_columns
    
    
    # 数据预处理函数
    def preprocess_data(X):
        # 保留原始列名
        original_columns = X.columns
        # 使用SimpleImputer填充NaN值
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        # 替换 inf 和 -inf
        X[np.isinf(X)] = np.nan
        X = imputer.fit_transform(X)
        
        # 确保所有值在float64范围内
        X = np.clip(X, np.finfo(np.float64).min, np.finfo(np.float64).max)
        
        # 将结果转换为 DataFrame 并设置列名
        X = pd.DataFrame(X, columns=original_columns)
        
        return X
    
    
    # 辅助函数：尝试转换为数值，如果失败则返回原始值
    def convert_to_numeric_or_keep(value):
        try:
            return pd.to_numeric(value)
        except ValueError:
            return value