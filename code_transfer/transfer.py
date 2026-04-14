import os
import pandas as pd

def load_csv_files_to_dict(directory):
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    data_dict = {}
    
    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        
        # 将 DataFrame 转换为字典
        file_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        # print("-----------------------------------------")
        
        data_dict[df.columns[0]] = file_dict
        # print(data_dict)
    
    return data_dict

# 示例使用
directory = 'resource/encode_table_16/CH4-C loss (%)'  # 替换为CSV文件所在的实际目录
data_dict = load_csv_files_to_dict(directory)

# # 打印结果验证
# for key, value in data_dict.items():
#     print(f"code_column: {key}")
#     print(f"Content: {value}\n")


def convert_csv_files(directory, conversion_dict, csv_dir, output_name):
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv') or f.endswith('.csv')]
    print('>>>=======================================<<<')
    print(csv_files)
    print('>>>=======================================<<<')
    ###############################################################################
    print('>>>=======================================<<<')
    print(conversion_dict)
    print('>>>=======================================<<<')
    for file in csv_files:
        file_path = os.path.join(csv_dir, file)
        df = pd.read_csv(file_path)
        print('>>>=======================================<<<')
        print(df.head())
        print('>>>=======================================<<<')
        for column, conver_dict in conversion_dict.items():
            # 创建一个反转字典，键和值交换
            reverse_dict = {v: k for k, v in conver_dict.items()}
            # 转换值
            df[column] = df[column].map(reverse_dict).fillna(df[column])
            pass    
        matrial_main = df['Material_Main'].tolist()[0]
        # 保存转换后的文件
        dirs = directory + output_name
        os.makedirs(dirs, exist_ok=True)
        converted_file_path = os.path.join(dirs, f"{matrial_main}.csv")
        df.to_csv(converted_file_path, index=False)

# 示例使用
output_name = '/output/1009' # 修改这里
csv_dir = r'output/Ga/raw_useall_True_1008/GaReasult' # 修改这里
directory = r'code_transfer'  # 替换为Excel文件所在的实际目录
convert_csv_files(directory, data_dict, csv_dir, output_name)