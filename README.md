# Code for Composting strategy optimization paper

## Quick Start

### 使用说明

* python == 3.7.13
* ```
  conda create -n cso python==3.7.13
  conda activate cso
  // 进入主目录路径下
  pip install -r requirements.txt
  ```
* 然后依次run all 数据分析.ipynb、模型训练.ipynb、全球预测.ipynb 即可，需修改每个文件的输入输出路径

### 文件说明

- data中包含原始数据以及数据处理后的文件
- output是输出的模型以及绘制的图，以及各个模型的表现情况
  - 其中Model_{target}文件夹保存了模型的训练参数。
- 数据分析.ipynb（data_processing.py）：包含了数据的分析与特征处理
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
