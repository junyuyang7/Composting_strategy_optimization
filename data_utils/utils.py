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
    
    # 填充缺失值
    def fill_missing(df, numerical_fea, categorical_fea, fillcontent=0, n_neighbors=10):
        """
        填充数据框中的缺失值，数值特征使用KNN填充，非数值特征使用众数填充。

        参数：
        - df: 数据框
        - numerical_fea: 数值特征列列表
        - categorical_fea: 非数值特征列列表
        - fillcontent: 填充数值特征的默认值（未使用）
        - n_neighbors: KNN填充器的邻居数量

        返回：
        - 填充后的数据框
        """
        
        # 处理数值特征列
        if numerical_fea:
            df_numerical = df[numerical_fea]
            
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
            df_filled = pd.DataFrame(df_filled, columns=numerical_fea)
            
            # 将填充后的数值特征列合并回原数据框
            df.update(df_filled)
        
        # 处理非数值特征列
        if categorical_fea:
            df_categorical = df[categorical_fea]
            
            # 初始化众数填充器
            imputer = SimpleImputer(strategy='most_frequent')
            
            # 对数据进行填充
            df_filled = imputer.fit_transform(df_categorical)
            
            # 将填充后的数据重新放回DataFrame
            df_filled = pd.DataFrame(df_filled, columns=categorical_fea)
            
            # 将填充后的非数值特征列合并回原数据框
            df.update(df_filled)
        
        return df
        # df[numerical_fea] = df[numerical_fea].fillna(df[numerical_fea].mean())
    
    
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
                    mean_encoded_values = df_nonan.groupby(cat)[trg].mean()
                    print(mean_encoded_values)
                    df[f'{cat}_{trg}'] = df[cat].map(mean_encoded_values)
                except Exception as e:
                    print('mean_encoding出现错误：', e)
                    print('cat:', cat, 'trg:', trg)
        cols = [col for col in df.columns if col not in target_feature]
        return df[cols]
    
   # 定义一个函数来使用KNN标准化去除异常值
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
        data_all.replace('-', 0, inplace=True)
        if data_all is None:
            raise ValueError("Failed to read the input file with provided encodings.")
    
        # # 初始化数值列和非数值列的列表
        # numerical_columns = []
        # non_numerical_columns = []
        
        # # 分类数值列和非数值列
        # for column in data_all.columns:
        #     try:
        #         # 过滤非空值
        #         non_empty_values = data_all[column].dropna()
        #         # 尝试将非空值转换为数值类型
        #         pd.to_numeric(non_empty_values)
        #         numerical_columns.append(column)
        #     except ValueError:
        #         non_numerical_columns.append(column)
        # print('num', numerical_columns)
        # print('non num', non_numerical_columns)
        # return data_all, numerical_columns, non_numerical_columns
    
        # 初始化数值列和非数值列的列表
        numerical_columns = []
        non_numerical_columns = []

        for column in data_all.columns:
            try:
                # 过滤非空值
                non_empty_values = data_all[column].dropna()
                # 尝试将非空值转换为数值类型
                converted_values = pd.to_numeric(non_empty_values, errors='coerce')
                
                if converted_values.notna().all():
                    print(column)
                    print((converted_values % 1)[converted_values % 1 != 0])
                    if (converted_values % 1 == 0).all():
                        non_numerical_columns.append(column)  # 整数类型视为非数值类型
                    else:
                        numerical_columns.append(column)
                else:
                    non_numerical_columns.append(column)
            except ValueError:
                non_numerical_columns.append(column)

        print('Numerical columns:', numerical_columns)
        print('Non-numerical (including integer) columns:', non_numerical_columns)
        return data_all, numerical_columns, non_numerical_columns
    
    
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