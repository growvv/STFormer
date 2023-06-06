## 代码模块
`config`：包含配置文件

`script`：包含运行脚本

`simvp/api`：包含一个实验的启动器

`simvp/core`：包含核心的运行插件和基准测试

`simvp/dataset`：包含数据集和数据加载器

`simvp/method`：包含simvp方法的实现

`simvp/model`：包含STFormer模型的实现

`simvp/modules`：包含核心组件的实现

`simvp/utils`：包含工具函数

## 数据集
适用于任何时空数据

例如：
2000条雷达回波数据集，
格式`T x C x H x W`，其中`T`为时间维度，`H`为高度维度，`W`为宽度维度, `C`为通道维度，`C=1`，表示仅雷达。


## 配置环境

1. 创建anaconda环境

`conda create -n simvp python=3.8`

2. 安装pytorch

`conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`

需要与系统cuda版本对应

3. 安装其他依赖

`pip install -r requirements.txt`

## 运行

1. 修改train_radar_gsta.sh，设置数据集路径、模型保存路径等

2. 运行 

`bash script/train_radar_gsta.sh`

## 推理

1. 修改infer_radar_gsta.sh，设置权重路径、数据集路径等

2. 运行

`bash script/infer_radar_gsta.sh`