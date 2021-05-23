# clstorch

## 摘要

基于numpy实现简易版深度学习框架clstorch，搭建神经网络时间MNIST手写数字识别，达到95%以上的准确率，并探究了神经网络的各个组件对最终结果的影响。

## 文件说明

* clstorch.py 基于numpy定义的简易框架
* dataloader.py transformpy 读取输入并进行相应的转换
* main.py 使用MLP进行手写数字识别的主程序
* dropout.py 引入dropout之后的模型
* finetune.py 对训练过的模型使用最小二乘法进行微调
* /dump 中间结果输出

## 训练曲线

![](https://github.com/TrueNobility303/clstorch/tree/master/dump)
