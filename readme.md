# Long Text Split

长文本切分模型，使用bert对超长文本进行切分。

## Feature

1. 使用现代化的trainer api，提供丰富的训练配置，例如梯度累积，学习率衰减等。
2. 提供蒸馏工作流，方便实现模型压缩和上线。
3. 提供评估示例接口，便于在流程中快速查看实验结果。
4. 提供简易gradio页面，快速展示示例。

## Usage

1. 安装依赖：pip install -r requirements.txt
2. 配置脚本，填写你的数据路径和模型路径
    1. 训练脚本run.sh
    2. 蒸馏脚本distil.sh
3. 开始训练：bash scripts/run.sh -n 1 -g 1

其中-n表示你有几台机器，-g表示每台机器有多少张卡，如果需要调试，可以加上-d 1。

## Data

数据组成示例：

{
    "text": "xxxx",
    "label": [0, 1, 0, 1]
}

其中text包含分隔符，label是分隔符对应的标签。