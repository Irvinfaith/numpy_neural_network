# -*- coding: utf-8 -*-
"""
Created on 2021/1/6 16:38

@author: Irvinfaith

@email: Irvinfaith@hotmail.com
"""

# from base_neural_network import NeuralNetwork
from core.base_neural_network import NeuralNetwork
from core.optimizer import *

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 导入样例数据
data_loader = load_iris()
data = data_loader['data']
# 进行归一化
mms = MinMaxScaler()
data = mms.fit_transform(data)
# 拆分训练测试集
X_train, X_test, y_train, y_test = train_test_split(data, data_loader['target'], test_size=0.3, random_state=101)

# 输入层
nn = NeuralNetwork(4, False)
# 添加全连接层，并定义神经元个数以及该层的激活函数
nn.add_dense_layer(16, "sigmoid")
nn.add_dense_layer(16, "sigmoid")
# 应用dropout，并设定留存率
# nn.add_dropout_layer(0.8)
# 添加输出层，并定义神经元个数以及该层的激活函数
nn.add_output_layer(3, "softmax")

# 定义损失函数：支持以下所有的激活函数：
# optimizer = AdaGrad(alpha=0.01)
# optimizer = AdaDelta(alpha=1, beta=0.95)
# optimizer = RMSProp(alpha=0.001, beta=0.9)
optimizer = Adam(alpha=0.05, beta_1=0.9, beta_2=0.99)
# optimizer = Adamax(alpha=0.05, beta_1=0.9, beta_2=0.99)
# optimizer = SGD(alpha=0.05, beta=0.99)
# optimizer = SGD(alpha=0.05)
# optimizer = BGD(alpha=0.05)
# 训练网络
nn.fit(X_train, y_train, epoch=100, batch_size=32, optimizer=optimizer, loss="cross_entropy_loss")
# 预测标签
prediction_y = nn.predict(X_test)
# 预测概率
prediction_proba = nn.predict_proba(X_test)
# 输出每一层网络的信息
dense_layer_info = nn.layer_info
