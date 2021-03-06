本项目的CSDN博客链接：https://blog.csdn.net/weixin_41578567/article/details/111482022

# 1. 概览

本项目主要用于神经网络的学习，通过基于numpy的实现，了解神经网络底层前向传播、反向传播以及各类优化器的原理。

该项目目前已实现的功能：

- 自定义多层的全连接层，并可定义多种激活函数
  - sigmoid
  - tanh
  - relu
  - softmax
- 定义dropout
- 支持多分类任务
- 支持多种优化器
  - SGD
  - BSGD
  - SGD with Momentum
  - AdaGrad
  - AdaDelta
  - RMSProp
  - Adam
  - AdaMax

# 2. Todo list

- [ ] 加入validation

- [x] 支持多分类任务

- [ ] 加入卷积层（CNN）
- [ ] 加入池化层
- [ ] 加入循环网络（RNN）



# 3. 运行

直接执行`main_binary_classification.py`，即可执行二分类问题的训练和预测

执行`main_multi_classification.py`，即可执行多分类问题的训练和预测


```shell
python main_binary_classification.py

"""
Epoch 1/100 - loss: 0.19804074649934453 - acc: 0.7462311557788944
Epoch 10/100 - loss: 0.05641219571461576 - acc: 0.9447236180904522
Epoch 20/100 - loss: 0.03296407980222495 - acc: 0.9698492462311558
Epoch 30/100 - loss: 0.024833182907967224 - acc: 0.9798994974874372
Epoch 40/100 - loss: 0.02147093826055232 - acc: 0.9824120603015075
Epoch 50/100 - loss: 0.018468433700412783 - acc: 0.9849246231155779
Epoch 60/100 - loss: 0.017207182621404478 - acc: 0.9849246231155779
Epoch 70/100 - loss: 0.016608808691509016 - acc: 0.9824120603015075
Epoch 80/100 - loss: 0.014850895654103682 - acc: 0.9849246231155779
Epoch 90/100 - loss: 0.014590506876720767 - acc: 0.9824120603015075
Epoch 100/100 - loss: 0.013794675383962845 - acc: 0.9874371859296482
"""
```

