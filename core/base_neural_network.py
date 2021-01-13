# -*- coding: utf-8 -*-
"""
Created on 2020/12/18 14:47

@author: Irvinfaith

@email: Irvinfaith@hotmail.com
"""
from collections import OrderedDict
import numpy as np
from utils.utils import series_to_array, array1d_to_onehot
from . import activation
from . import cost_function
import warnings

warnings.filterwarnings("once")


class NeuralNetwork:
    """
    自定义神经网络，初始化需定义输入的神经元个数，以及是否启用偏置项
    """

    def __init__(self, feature_num, bias=True):
        self.feature_num = feature_num
        self.bias = bias
        self.layer_index = 0
        self.layer_info = OrderedDict()
        self.batch_info = {}
        self.num_classes = 2
        self.is_multi = False

    @staticmethod
    def activation_function(method, x, derive=False):
        return getattr(activation, method)(x, derive=derive)

    @staticmethod
    def loss_function(method, true_y, prediction_y, derive=False):
        return getattr(cost_function, method)(true_y, prediction_y, derive=derive)

    def add_dense_layer(self, neural_num, activation):
        """全连接层

        Args:
            neural_num: 神经元个数
            activation: 激活函数

        Returns:

        """
        self.layer_index += 1
        if self.bias:
            self.layer_info[self.layer_index] = {
                "ori": np.random.randn(self.feature_num + 1, neural_num + 1) if self.layer_index == 1 else
                np.random.randn(self.layer_info[self.layer_index - 1]['ori'].shape[1], neural_num + 1)}
        else:
            self.layer_info[self.layer_index] = {
                "ori": np.random.randn(self.feature_num, neural_num) if self.layer_index == 1 else
                np.random.randn(self.layer_info[self.layer_index - 1]['ori'].shape[1], neural_num)}
        self.layer_info[self.layer_index].update({"activation_function": activation, 'layer_type': 'dense'})

    def add_dropout_layer(self, rate):
        """Drop out 层

        Args:
            rate: 失活率

        Returns:

        """
        self.layer_info[self.layer_index].update({"layer_type": 'dropout', "drop_out_rate": rate})

    def add_output_layer(self, neural_num, activation):
        """输出层

        Args:
            neural_num: 输出神经元个数，默认1
            activation: 激活函数

        Returns:

        """
        self.layer_index += 1
        self.layer_info[self.layer_index] = {
            "ori": np.random.randn(self.layer_info[self.layer_index - 1]['ori'].shape[1], neural_num),
            "activation_function": activation,
            "layer_type": "dense"
        }

    def fit(self, X, y, optimizer, loss="mse", epoch=50, batch_size=64, threshold=0.5):
        """

        Args:
            X: 输入X
            y: 输入y
            optimizer: 优化器
            loss: 损失函数
            epoch: 优化次数
            batch_size: batch数
            threshold: 判定为正的阈值

        Returns:

        """

        self.num_classes = np.unique(y).shape[0]
        self.is_multi = False
        # X shape : (m, n)
        X = series_to_array(X)
        y = series_to_array(y)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if self.num_classes > 2:
            self.is_multi = True
            y = array1d_to_onehot(y, self.num_classes)
        if self.bias:
            # 为输入x添加偏置项，默认都为1
            X = np.c_[X, np.ones(X.shape[0])]
        # 赋初始权重值
        for layer_index in self.layer_info.keys():
            self.layer_info[layer_index]['new'] = self.layer_info[layer_index]['ori']
        for i in range(epoch):
            self._fit_one_epoch(X, y, optimizer, loss, batch_size)
            # 预测结果
            prediction_proba = self._predict_proba(X, bias=False)
            prediction = self._predict(X, threshold, bias=False)
            # 计算损失
            total_loss = self.loss_function(loss, y, prediction_proba)
            # backprogation
            accuracy = self.calc_accuracy(y, prediction)
            if (i+1) % 10 == 0:
                print(f"Epoch {i + 1}/{epoch} - loss: {total_loss} - acc: {accuracy}")

    def _fit_one_epoch(self, x, y, optimizer, loss, batch_size):
        """

        Args:
            x: 输入x
            y: 输入y
            optimizer: 优化器
            batch_size: batch数量

        Returns:

        """
        # 若是多分类，需将y转为onehot
        # if self.is_multi:
        #     y = target_to_onehot(y, self.num_classes)

        # 批量梯度下降
        if optimizer.name == "BGD" and batch_size is not None:
            # warnings.warn("When `BGD`, there is no need to set `batch_size`, since the batch is the whole samples")
            m = x.shape[0]
            # 轮次从1开始，防止计算Adam的时候初始轮次为0导致除数为0
            for _ in range(1, m):
                self._fit_one_round(x, y, optimizer, loss, _)
        elif optimizer.name == "SGD" and batch_size is not None:
            # warnings.warn("Normally `SGD` refers to there is only one sample in one round batch trainning. "
            #      "In this scenario, if you set a `batch_size` when the optimizer is `SGD`, it is actually called"
            #      "`mini-batch gradient descent` aka `MBGD`")
            m = x.shape[0]
            # index = np.random.choice(np.arange(m), batch_size, replace=False)
            # 轮次从1开始，防止计算Adam的时候初始轮次为0导致除数为0
            # for _, data in enumerate(zip(x[index, :], y[index, :]), start=1):
            #     self._fit_one_round(data[0], data[1], optimizer, loss, _)
            batch_round = int(m // batch_size)
            # 轮次从1开始，防止计算Adam的时候初始轮次为0导致除数为0
            for _ in range(1, batch_round + 1):
                index = np.random.choice(np.arange(m), batch_size, replace=False)
                self._fit_one_round(x[index, :], y[index, :], optimizer, loss, _)
        elif optimizer.name == "SGD" and batch_size is None:
            # 采用SGD，即每次epoch抽取1个样本进行approch
            # 训练速度快，但是不一定是全局最优，损失函数会有震荡
            _shuffle_index = np.random.permutation(x.shape[0])
            for _, data in enumerate(zip(x, y), start=1):
                self._fit_one_round(data[0][_shuffle_index], data[1][_shuffle_index], optimizer, loss, _)
        elif batch_size:
            m = x.shape[0]
            batch_round = int(m // batch_size)
            # 轮次从1开始，防止计算Adam的时候初始轮次为0导致除数为0
            for _ in range(1, batch_round + 1):
                index = np.random.choice(np.arange(m), batch_size, replace=False)
                self._fit_one_round(x[index, :], y[index, :], optimizer, loss, _)

    def _fit_one_round(self, x, y, optimizer, loss, i=None):
        """

        Args:
            x: 输入x
            y: 输入y
            optimizer: 优化器
            loss: 损失函数
            i: 优化轮次

        Returns:

        """
        # 转为2维矩阵 (n,) -> (1,n)
        x = np.atleast_2d(x)
        optimizer_info = OrderedDict()

        # 前向传播
        for layer_index in self.layer_info.keys():
            if layer_index == 1:
                out = np.dot(x, self.layer_info[layer_index]['new'])
            else:
                # if self.layer_info[layer_index]['layer_type'] == 'dense':
                out = np.dot(optimizer_info[layer_index - 1]['activation'],
                             self.layer_info[layer_index]['new'])

            # 激活函数
            # activation = self.sig_deriv(out)
            activation = self.activation_function(self.layer_info[layer_index]['activation_function'],
                                                  out)
            if self.layer_info[layer_index]['layer_type'] == 'dropout':
                # 若为dropout，前一层的激活输出 * dropout层的失活mask / 失活概率
                drop_out_rate = self.layer_info[layer_index]['drop_out_rate']
                drop_out_mask = np.random.binomial(1, drop_out_rate, size=activation.shape)
                activation = np.multiply(activation, drop_out_mask) / drop_out_rate
            optimizer_info[layer_index] = {'out': out, 'activation': activation}

        # 反向传播
        # 从后向前更新
        for layer_index in reversed(self.layer_info.keys()):
            if layer_index == self.layer_index:
                # 计算误差
                error = self.loss_function(loss, y, optimizer_info[list(optimizer_info.keys())[-1]]['activation'],
                                           derive=True)

                if self.is_multi:
                    derive = error
                else:
                    this_out_deriv = self.activation_function(self.layer_info[layer_index][
                                                                  'activation_function'],
                                                              optimizer_info[layer_index]['out'],
                                                              True)
                    # 输出层的偏导, shape: (m, next_n=1)
                    derive = np.multiply(error, this_out_deriv)
                optimizer_info[layer_index].update({'derive': derive})
                # last_activation: 上一层的激活输出，shape: (m, this_n)
                last_activation = optimizer_info[layer_index - 1]['activation']

                # weight shape: (this_n, next_n=1)
                # 计算梯度gradient
                gradient = last_activation.T.dot(derive)
                # 执行优化器，更新参数
                self.cal_optimizer(optimizer, gradient, layer_index, i, optimizer_info)
            elif layer_index == 1:
                # last_derive: 前一层传递的偏导，shape: (m, next_n下一层神经元个数)
                last_derive = optimizer_info[layer_index + 1]['derive']
                # last_ori_weight: 上一层的权重矩阵，shape: (this_n, next_n) (当前层神经元个数, 下层神经元个数)
                last_ori_weight = self.layer_info[layer_index + 1]['ori']
                # this_out_deriv: 该层输出层的偏导，shape: (m, this_n)
                # this_out_deriv = self.sig_deriv(optimizer_info[layer_index]['out'], True)
                this_out_deriv = self.activation_function(self.layer_info[layer_index][
                                                              'activation_function'],
                                                          optimizer_info[layer_index]['out'],
                                                          True)
                # 接收前面层传递的链式偏导并继续往下层传递, shape: (m, next_n)
                derive = np.multiply(last_derive.dot(last_ori_weight.T), this_out_deriv)
                optimizer_info[layer_index].update({'derive': derive})
                # 计算梯度gradient：
                gradient = x.T.dot(derive)
                # 执行优化器，更新参数
                self.cal_optimizer(optimizer, gradient, layer_index, i, optimizer_info)
            else:
                # last_derive: 前一层传递的偏导，shape: (m, next_n下一层神经元个数)
                last_derive = optimizer_info[layer_index + 1]['derive']
                # last_ori_weight: 上一层的权重矩阵，shape: (this_n, next_n) (当前层神经元个数, 下层神经元个数)
                last_ori_weight = self.layer_info[layer_index + 1]['ori']
                # this_out_deriv: 该层输出层的偏导，shape: (m, this_n)
                # this_out_deriv = self.sig_deriv(optimizer_info[layer_index]['out'], True)
                this_out_deriv = self.activation_function(self.layer_info[layer_index][
                                                              'activation_function'],
                                                          optimizer_info[layer_index]['out'],
                                                          True)
                # 接收前面层传递的链式偏导并继续往下层传递, shape: (m, next_n)
                derive = np.multiply(last_derive.dot(last_ori_weight.T), this_out_deriv)
                optimizer_info[layer_index].update({'derive': derive})
                # last_activation: 上一层的激活输出，shape: (m, this_n)
                last_activation = optimizer_info[layer_index - 1]['activation']

                # Optimizer: 计算梯度gradient，shape：(this_n, next_n)
                gradient = last_activation.T.dot(derive)
                # 执行优化器，更新参数
                self.cal_optimizer(optimizer, gradient, layer_index, i, optimizer_info)
        self.batch_info[i] = optimizer_info

    def cal_optimizer(self, optimizer, gradient, layer_index, i, tmp_value):
        """执行优化器，更新参数

        Args:
            optimizer: 优化器对象
            gradient: 上一轮的梯度
            layer_index: 网络层
            i: 优化轮次
            tmp_value: 优化中间数据字典

        Returns:

        """
        if optimizer.name == "SGD":
            # 判断是否计算动量
            if optimizer.beta:
                # 初始化动量momentum
                if i == 1:
                    last_momentum = 0
                else:
                    last_momentum = self.batch_info[i - 1][layer_index]["momentum"]
                this_momentum = optimizer.calc_momentum(gradient, last_momentum)
                tmp_value[layer_index].update({"momentum": this_momentum})
                self.layer_info[layer_index]['new'] = optimizer.update_target(
                    self.layer_info[layer_index]['new'], gradient, this_momentum)
            else:
                self.layer_info[layer_index]['new'] = optimizer.update_target(
                    self.layer_info[layer_index]['new'], gradient)
        elif optimizer.name == "AdaGrad":
            if i == 1:
                total_squared_gradient_sum = np.zeros(gradient.shape)
            else:
                total_squared_gradient_sum = self.batch_info[i - 1][layer_index]["total_gradient_squared_sum"]
            total_squared_gradient_sum += np.power(gradient, 2)
            tmp_value[layer_index].update({"total_gradient_squared_sum": total_squared_gradient_sum})
            self.layer_info[layer_index]['new'] = optimizer.update_target(
                self.layer_info[layer_index]['new'], gradient, total_squared_gradient_sum)
        elif optimizer.name == "AdaDelta":
            if i == 1:
                last_ewa_squared_gradient = np.zeros(gradient.shape)
                last_ewa_squared_delta_x = 0
            else:
                last_ewa_squared_gradient = self.batch_info[i - 1][layer_index]["ewa_squared_gradient"]
                last_ewa_squared_delta_x = self.batch_info[i - 1][layer_index]["ewa_squared_delta_x"]
            ewa_squared_gradient = optimizer.calc_ewa_squared_value(gradient, last_ewa_squared_gradient)
            rms_gradient = optimizer.calc_rms_value(ewa_squared_gradient)
            last_rms_delta_x = optimizer.calc_rms_value(last_ewa_squared_delta_x)
            delta_x = optimizer.calc_delta_x(last_rms_delta_x, rms_gradient, gradient)
            ewa_squared_delta_x = optimizer.calc_ewa_squared_value(delta_x, last_ewa_squared_delta_x)
            tmp_value[layer_index].update(
                {"ewa_squared_gradient": ewa_squared_gradient, "ewa_squared_delta_x": ewa_squared_delta_x})
            self.layer_info[layer_index]['new'] = optimizer.update_target(
                self.layer_info[layer_index]['new'], gradient, last_ewa_squared_gradient,
                last_ewa_squared_delta_x)
        elif optimizer.name == "RMSProp":
            if i == 1:
                last_ewa_squared_gradient = 0
            else:
                last_ewa_squared_gradient = self.batch_info[i - 1][layer_index]["ewa_squared_gradient"]
            ewa_squared_gradient = optimizer.calc_ewa_squared_value(gradient, last_ewa_squared_gradient)
            tmp_value[layer_index].update({"ewa_squared_gradient": ewa_squared_gradient})
            self.layer_info[layer_index]['new'] = optimizer.update_target(
                self.layer_info[layer_index]['new'], gradient, last_ewa_squared_gradient)
        elif optimizer.name == "Adam":
            if i == 1:
                last_momentum = 0
                last_ewa_squared_gradient = 0
            else:
                last_momentum = self.batch_info[i - 1][layer_index]["momentum"]
                last_ewa_squared_gradient = self.batch_info[i - 1][layer_index]["ewa_squared_gradient"]
            this_momentum = optimizer.calc_momentum(gradient, last_momentum)
            ewa_squared_gradient = optimizer.calc_ewa_squared_value(gradient, last_ewa_squared_gradient)
            tmp_value[layer_index].update({"momentum": this_momentum, "ewa_squared_gradient": ewa_squared_gradient})
            self.layer_info[layer_index]['new'] = optimizer.update_target(
                self.layer_info[layer_index]['new'], gradient, last_momentum, last_ewa_squared_gradient, i)
        elif optimizer.name == "Adamax":
            if i == 1:
                last_momentum = 0
                last_ewa_squared_gradient = 0
            else:
                last_momentum = self.batch_info[i - 1][layer_index]["momentum"]
                last_ewa_squared_gradient = self.batch_info[i - 1][layer_index]["ewa_squared_gradient"]
            this_momentum = optimizer.calc_momentum(gradient, last_momentum)
            ewa_squared_gradient = optimizer.calc_ewa_squared_value(gradient, last_ewa_squared_gradient)
            tmp_value[layer_index].update(
                {"momentum": this_momentum, "ewa_squared_gradient": ewa_squared_gradient})
            self.layer_info[layer_index]['new'] = optimizer.update_target(
                self.layer_info[layer_index]['new'], gradient, last_momentum, last_ewa_squared_gradient, i)

    def calc_accuracy(self, y_true, y_prediciton):
        if len(y_true.shape) != 1 and not self.is_multi:
            y_true = y_true.reshape(1, -1)[0]
        if self.is_multi:
            true_count = sum(y_true.argmax(axis=1) == y_prediciton)
            accuracy = true_count / y_true.shape[0]
            # print(accuracy)
            return accuracy
        if len(y_prediciton.shape) != 1:
            y_prediciton = y_prediciton.reshape(1, -1)[0]
        true_count = sum(y_true == y_prediciton)
        accuracy = true_count / y_true.shape[0]
        # print(accuracy)
        return accuracy

    def predict(self, X, threshold=0.5):
        return self._predict(X, threshold, self.bias).reshape(1, -1)[0, :]

    def predict_proba(self, X):
        return self._predict_proba(X, self.bias).reshape(1, -1)[0, :]

    def _predict(self, X, threshold, bias=False):
        prediction_proba = self._predict_proba(X, bias)
        if self.is_multi:
            return prediction_proba.argmax(axis=1)
        return np.where(prediction_proba > threshold, 1, 0)

    def _predict_proba(self, X, bias=False):
        x = np.atleast_2d(X)
        if bias:
            x = np.c_[x, np.ones((x.shape[0]))]
        out = x
        for layer_index in self.layer_info.keys():
            o1 = np.dot(out, self.layer_info[layer_index]['new'])
            # out = self.sig_deriv(o1)
            out = self.activation_function(self.layer_info[layer_index]['activation_function'], o1)
        return out
