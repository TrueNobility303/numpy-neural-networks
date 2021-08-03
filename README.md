# clstorch

## Abstract

Neural netwroks with Numpy for handwritten digit recognition on MNIST dataset, getting an accuracy of above 95%.
Ablation studies are done! Refer to the report!

## Files

* clstorch.py: neural networks with Numpy
* dataloader.py: codes of dataloder
* main.py: the main program
* dropout.py: network with dropout
* finetune.py: codes for finetuning a model
* /dump: directory for outputs
* project.pdf: report

## Training Curve

![](https://github.com/TrueNobility303/clstorch/blob/master/dump/loss-valid.png)

## 摘要

基于numpy实现简易版深度学习框架clstorch，搭建神经网络时间MNIST手写数字识别，达到95%以上的准确率，并探究了神经网络的各个组件对最终结果的影响。

## 文件说明

* clstorch.py 基于numpy定义的简易框架
* dataloader.py transformpy 读取输入并进行相应的转换
* main.py 使用MLP进行手写数字识别的主程序
* dropout.py 引入dropout之后的模型
* finetune.py 对训练过的模型使用最小二乘法进行微调
* /dump 中间结果输出
* project.pdf 报告，代码详细说明与实验

## 训练曲线

![](https://github.com/TrueNobility303/clstorch/blob/master/dump/loss-valid.png)
