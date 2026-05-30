# Code for Composting strategy optimization paper

## Quick Start

### 使用说明

- python == 3.9.25
```python
  conda create -n cso python==3.9.25
  conda activate cso
  # 进入主目录路径下
  pip install -r requirements.txt
  # 模型训练 
  python training.py
  # 
```
-  
- 

### 文件说明

- model 文件包含了预测使用的模型，如果添加模型可继承其中的ModelBase类进行添加使用
- data中包含原始数据以及数据处理后的文件
- output是输出的模型以及绘制的图，以及各个模型的表现情况
  - 其中Model_{target}文件夹保存了模型的训练参数。
- 数据分析.ipynb（data_processing.py）：包含了数据的分析与特征处理，包括对类别特征和数值特征的处理
- 模型训练.ipynb（training.py）：主要进行不同模型的训练
- 全球预测.ipynb（Final_predict.py）：主要进行最终的预测
- Tips：data_processing.py / training.py / Final_predict.py：这几个是为了导出依赖从而创建的
- ```
  // 可以使用pipreqs导出只与本项目相关的依赖
  pip install pipreqs
  pipreqs ./ --encoding='iso-8859-1'
  ```



### Docker（之后考虑使用）

**使用docker**

* docker安装教程：
  * 安装hyper-v：[Win11 家庭版/专业版开启Hyper-V - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/577980646)
  * window11 家庭版 安装docker：[超详细Windows11家庭中文版系统安装Docker-20230401\_windows11安装docker-CSDN博客](https://blog.csdn.net/m0_37802038/article/details/129893827)
  * linux 安装docker：[Ubuntu Docker 安装 | 菜鸟教程 (runoob.com)](https://www.runoob.com/docker/ubuntu-docker-install.html)
* docker使用教程：
  * [Docker最新超详细版教程通俗易懂(基础版) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/442442997)
  * 视频教程：https://www.bilibili.com/video/BV11L411g7U1/
  * 视频教程文字版：https://docker.easydoc.net/
 
### Git使用
- 如何合并分支：https://blog.csdn.net/m0_57236802/article/details/133826681
- 更新本地代码：https://www.cnblogs.com/delav/p/11118555.html

### 遗传算法优化代码的使用
- 1.先创建cso环境
 - conda create -n cso python==3.7.13
 - conda activate cso
 - pip install -r requirements.txt
 - 可能会有缺失的包，缺什么再补啥吧

- 2.GaOptimization_NSGAII.py
  - GaOptimization_NSGAII.py line10：修改输出文件夹（改一下最后日期就行了1008）
  - GaOptimization_NSGAII.py line11/12：不用管
  - GaOptimization_NSGAII.py line22：设置 num_runs 轮数
  - 修改完后直接运行命令行 "python GaOptimization_NSGAII.py"
  - 会提示输入Material_Main种类（输入0-10中的数字即可，对应的字典在resource_mean/encode_table_16/CO2-C loss (%)/Material_Main_ordinal_encoding.csv）
  - 可以开启11个终端界面同时进行优化

- 3.code_transfer/transfer.py（将数字编码转化回去，例如Material_Main=0 对应Material_Main="Cattle manure"）
  - transfer.py line60：修改输出文件夹
  - transfer.py line61：与GaOptimization_NSGAII.py line10中的保持一致即可
  - 最终输出的文件在 code_transfer/{$transfer.py line60} 的路径中
