# -*- coding: utf-8 -*-
"""
Created on 2020/12/18 14:47

@author: Irvinfaith

@email: Irvinfaith@hotmail.com
"""
from collections import OrderedDict
from warnings import warn

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


class NeuralNetwork:
    """
    自定义神经网络，初始化需定义输入的神经元个数，以及是否启用偏置项
    """
    def __init__(self, feature_num, bias=True):
        self.num_input = feature_num
        self.bias = bias
        self.input_vertex_list = ['x' + str(i) for i in range(1, self.num_input + 1)]
        if bias:
            self.input_vertex_list += ['b']
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.input_vertex_list)
        self.layer_num = 0
        self.dense_layer_vertex = OrderedDict()
        self.dense_layer_info = OrderedDict()
        self.batch_info = {}

    @staticmethod
    def activation_function(method, x, derive=False):
        return getattr(__import__("activation"), method)(x, derive=derive)

    @staticmethod
    def loss_function(method, true_y, prediction_y, derive=False):
        return getattr(__import__("cost_function"), method)(true_y, prediction_y, derive=derive)


    def add_dropout_layer(self, rate):
        """Drop out 层

        Args:
            rate: 失活率

        Returns:

        """
        self.dense_layer_info[self.layer_num].update({"layer_type": 'dropout', "drop_out_rate": rate})

    def add_dense_layer(self, num_neural, activation='sigmoid'):
        """全连接层

        Args:
            num_neural: 神经元个数
            activation: 激活函数

        Returns:

        """
        self.layer_num += 1
        self.dense_layer_vertex[self.layer_num] = [f'I{self.layer_num}_{i}' for i in range(1, num_neural + 1)]
        if self.bias:
            self.dense_layer_vertex[self.layer_num] += [f'b{self.layer_num}']
        self.G.add_nodes_from(self.dense_layer_vertex[self.layer_num])
        if self.layer_num <= 1:
            if self.bias:
                self.dense_layer_info[self.layer_num] = {
                    "ori": np.random.randn(self.num_input + 1, num_neural + 1)}
                self.G.add_edges_from(
                    [(self.input_vertex_list[-1], b) for b in self.dense_layer_vertex[self.layer_num][:-1]])
                self.G.add_edges_from(
                    [(a, b) for a in self.input_vertex_list[:-1] for b in self.dense_layer_vertex[self.layer_num]])
            else:
                self.dense_layer_info[self.layer_num] = {
                    "ori": np.random.randn(self.num_input, num_neural)}
                self.G.add_edges_from(
                    [(a, b) for a in self.input_vertex_list for b in self.dense_layer_vertex[self.layer_num]])
        else:
            if self.bias:
                self.dense_layer_info[self.layer_num] = {
                    "ori": np.random.randn(len(self.dense_layer_vertex[self.layer_num - 1]), num_neural + 1)}
                self.G.add_edges_from(
                    [(self.dense_layer_vertex[self.layer_num - 1][-1], b) for b in
                     self.dense_layer_vertex[self.layer_num][:-1]])
                self.G.add_edges_from([(a, b) for a in self.dense_layer_vertex[self.layer_num - 1][:-1] for b in
                                       self.dense_layer_vertex[self.layer_num][:-1]])
            else:
                self.dense_layer_info[self.layer_num] = {
                    "ori": np.random.randn(len(self.dense_layer_vertex[self.layer_num - 1]), num_neural)}
                self.G.add_edges_from([(a, b) for a in self.dense_layer_vertex[self.layer_num - 1] for b in
                                       self.dense_layer_vertex[self.layer_num]])
        self.dense_layer_info[self.layer_num].update({"activation_function": activation, 'layer_type': 'dense'})

    def add_output_layer(self, num_neural=1, activation="sigmoid"):
        """输出层

        Args:
            num_neural: 输出神经元个数，默认1
            activation: 激活函数

        Returns:

        """
        self.layer_num += 1
        self.dense_layer_vertex[self.layer_num] = [f'O{self.layer_num}_{i}' for i in range(1, num_neural + 1)]
        self.G.add_nodes_from(self.dense_layer_vertex[self.layer_num])
        self.dense_layer_info[self.layer_num] = {
            "ori": np.random.randn(len(self.dense_layer_vertex[self.layer_num - 1]), num_neural),
            "activation_function": activation,
            "layer_type": "dense"
        }
        self.G.add_edges_from([(a, b) for a in self.dense_layer_vertex[self.layer_num - 1] for b in
                               self.dense_layer_vertex[self.layer_num]])

    def _get_path_by_node(self, start, end):
        return [path for path in nx.all_simple_paths(self.G, start, end)]

    def _get_path_by_edge(self, start, end, path: str):
        # path 用 | 分割
        _out = []
        for _ in self._get_path_by_node(start, end):
            str_path = '|'.join(_)
            if path in str_path:
                _out.append(_)
        return _out

    def fit(self, X, y, optimizer, loss="mse", epoch=50, batch_size=64, threshold=0.5):
        """

        Args:
            X: 输入X
            y: 输入y
            optimizer: 优化器
            epoch: 优化次数
            batch_size: batch数
            threshold: 判定为正的阈值

        Returns:

        """
        import pandas
        self.num_classes = np.unique(y).shape[0]
        self.is_multi = False
        # X shape : (m, n)
        if isinstance(X, pandas.DataFrame):
            X = np.array(X)
        if isinstance(y, pandas.DataFrame) or isinstance(y, pandas.Series):
            y = np.array(y)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if self.num_classes > 2:
            self.is_multi = True
        if self.bias:
            # 为输入x添加偏置项，默认都为1
            X = np.c_[X, np.ones(X.shape[0])]
        # 赋初始权重值
        for layer_index in self.dense_layer_info.keys():
            self.dense_layer_info[layer_index]['new'] = self.dense_layer_info[layer_index]['ori']
        for i in range(epoch):

            self._fit_one_epoch(X, y, optimizer, loss, batch_size)
            # 预测结果
            prediction_proba = self._predict_proba(X, bias=False)
            prediction = self._predict(X, threshold, bias=False)
            # mse
            total_loss = self.loss_function(loss, y, prediction_proba)
            # total_loss = cost_function.mse(y, prediction_proba)
            # elif loss == "mae":

            # total_loss = 1 / 2 * np.sum((y - prediction_proba) ** 2)
            # backprogation
            accuracy = self.calc_accuracy(y, prediction)
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
        if self.is_multi:
            trans_y = np.zeros((y.shape[0], self.num_classes))
            for index, _ in enumerate(y):
                trans_y[index][_] += 1
            y = trans_y
        # 批量梯度下降
        if optimizer.name == "BGD" and batch_size is not None:
            raise Exception("When BGD, there is no need to set `batch_size`, since the batch is the whole samples")
        elif optimizer.name == "SGD" and batch_size is not None:
            warn("Normally `SGD` refers to there is only one sample in one round batch trainning. "
                 "In this scenario, if you set a `batch_size` when the optimizer is `SGD`, it is actually called"
                 "`mini-batch gradient descent` aka `MBGD`")
            m = x.shape[0]
            index = np.random.choice(np.arange(m), batch_size, replace=False)
            # 轮次从1开始，防止计算Adam的时候初始轮次为0导致除数为0
            for _, data in enumerate(zip(x[index, :], y[index, :]), start=1):
                self._fit_one_round(data[0], data[1], optimizer, loss, _)
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
            # index = np.random.choice(np.arange(m), batch_size, replace=False)
            # # 轮次从1开始，防止计算Adam的时候初始轮次为0导致除数为0
            # for _, data in enumerate(zip(x[index, :], y[index, :]), start=1):
            #     self._fit_one_round(data[0], data[1], optimizer, _)

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
        tmp_value = OrderedDict()

        # 前向传播
        for layer_index in self.dense_layer_info.keys():
            if layer_index == 1:
                out = np.dot(x, self.dense_layer_info[layer_index]['new'])
            else:
                # if self.dense_layer_info[layer_index]['layer_type'] == 'dense':
                out = np.dot(tmp_value[layer_index - 1]['activation'],
                             self.dense_layer_info[layer_index]['new'])

            # 激活函数
            # activation = self.sig_deriv(out)
            activation = self.activation_function(self.dense_layer_info[layer_index]['activation_function'],
                                                  out)
            if self.dense_layer_info[layer_index]['layer_type'] == 'dropout':
                # 若为dropout，前一层的激活输出 * dropout层的失活mask / 失活概率
                drop_out_rate = self.dense_layer_info[layer_index]['drop_out_rate']
                drop_out_mask = np.random.binomial(1, drop_out_rate, size=activation.shape)
                activation = np.multiply(activation, drop_out_mask) / drop_out_rate
            tmp_value[layer_index] = {'out': out, 'activation': activation}

        # 反向传播
        # 从后向前更新
        for layer_index in reversed(self.dense_layer_info.keys()):
            if layer_index == self.layer_num:
                # 计算误差
                # print(y[0])
                # _out = tmp_value[list(tmp_value.keys())[-1]]['activation']
                # print(_out[0])
                error = self.loss_function(loss, y, tmp_value[list(tmp_value.keys())[-1]]['activation'], derive=True)
                # print("error:", error)
                # error = tmp_value[list(tmp_value.keys())[-1]]['activation'] - y
                # this_out_deriv = self.sig_deriv(tmp_value[layer_index]['out'], True)
                this_out_deriv = self.activation_function(self.dense_layer_info[layer_index][
                                                              'activation_function'],
                                                          tmp_value[layer_index]['out'],
                                                          True)
                # 输出层的偏导, shape: (m, next_n=1)
                derive = np.multiply(error, this_out_deriv)
                tmp_value[layer_index].update({'derive': derive})
                # last_activation: 上一层的激活输出，shape: (m, this_n)
                last_activation = tmp_value[layer_index - 1]['activation']

                # weight shape: (this_n, next_n=1)
                # self.dense_layer_info[layer_index]['new'] -= alpha * last_activation.T.dot(derive)
                # 计算梯度gradient
                gradient = last_activation.T.dot(derive)
                # 执行优化器，更新参数
                self.cal_optimizer(optimizer, gradient, layer_index, i, tmp_value)
            elif layer_index == 1:
                # last_derive: 前一层传递的偏导，shape: (m, next_n下一层神经元个数)
                last_derive = tmp_value[layer_index + 1]['derive']
                # last_ori_weight: 上一层的权重矩阵，shape: (this_n, next_n) (当前层神经元个数, 下层神经元个数)
                last_ori_weight = self.dense_layer_info[layer_index + 1]['ori']
                # this_out_deriv: 该层输出层的偏导，shape: (m, this_n)
                # this_out_deriv = self.sig_deriv(tmp_value[layer_index]['out'], True)
                this_out_deriv = self.activation_function(self.dense_layer_info[layer_index][
                                                              'activation_function'],
                                                          tmp_value[layer_index]['out'],
                                                          True)
                # 接收前面层传递的链式偏导并继续往下层传递, shape: (m, next_n)
                derive = np.multiply(last_derive.dot(last_ori_weight.T), this_out_deriv)
                tmp_value[layer_index].update({'derive': derive})
                # 计算梯度gradient：
                gradient = x.T.dot(derive)
                # 执行优化器，更新参数
                self.cal_optimizer(optimizer, gradient, layer_index, i, tmp_value)
            else:
                # last_derive: 前一层传递的偏导，shape: (m, next_n下一层神经元个数)
                last_derive = tmp_value[layer_index + 1]['derive']
                # last_ori_weight: 上一层的权重矩阵，shape: (this_n, next_n) (当前层神经元个数, 下层神经元个数)
                last_ori_weight = self.dense_layer_info[layer_index + 1]['ori']
                # this_out_deriv: 该层输出层的偏导，shape: (m, this_n)
                # this_out_deriv = self.sig_deriv(tmp_value[layer_index]['out'], True)
                this_out_deriv = self.activation_function(self.dense_layer_info[layer_index][
                                                              'activation_function'],
                                                          tmp_value[layer_index]['out'],
                                                          True)
                # 接收前面层传递的链式偏导并继续往下层传递, shape: (m, next_n)
                derive = np.multiply(last_derive.dot(last_ori_weight.T), this_out_deriv)
                tmp_value[layer_index].update({'derive': derive})
                # last_activation: 上一层的激活输出，shape: (m, this_n)
                last_activation = tmp_value[layer_index - 1]['activation']

                # Optimizer: 计算梯度gradient，shape：(this_n, next_n)
                gradient = last_activation.T.dot(derive)
                # 执行优化器，更新参数
                self.cal_optimizer(optimizer, gradient, layer_index, i, tmp_value)
        self.batch_info[i] = tmp_value
        # # 更新字典中的权重
        # for layer_index in self.dense_layer_info.keys():
        #     self.dense_layer_info[layer_index]['ori'] = self.dense_layer_info[layer_index]['new']

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
                self.dense_layer_info[layer_index]['new'] = optimizer.update_target(
                    self.dense_layer_info[layer_index]['new'], gradient, this_momentum)
            else:
                self.dense_layer_info[layer_index]['new'] = optimizer.update_target(
                    self.dense_layer_info[layer_index]['new'], gradient)
        elif optimizer.name == "AdaGrad":
            if i == 1:
                total_squared_gradient_sum = np.zeros(gradient.shape)
            else:
                total_squared_gradient_sum = self.batch_info[i - 1][layer_index]["gradient_sum"]
            total_squared_gradient_sum += np.power(gradient, 2)
            tmp_value[layer_index].update({"total_gradient_squared_sum": total_squared_gradient_sum})
            self.dense_layer_info[layer_index]['new'] = optimizer.update_target(
                self.dense_layer_info[layer_index]['new'], gradient, total_squared_gradient_sum)
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
            tmp_value[layer_index].update({"ewa_squared_gradient": ewa_squared_gradient, "ewa_squared_delta_x": ewa_squared_delta_x})
            self.dense_layer_info[layer_index]['new'] = optimizer.update_target(
                self.dense_layer_info[layer_index]['new'], gradient, last_ewa_squared_gradient, last_ewa_squared_delta_x)
        elif optimizer.name == "RMSProp":
            if i == 1:
                last_ewa_squared_gradient = 0
            else:
                last_ewa_squared_gradient = self.batch_info[i - 1][layer_index]["ewa_squared_gradient"]
            ewa_squared_gradient = optimizer.calc_ewa_squared_value(gradient, last_ewa_squared_gradient)
            tmp_value[layer_index].update({"ewa_squared_gradient": ewa_squared_gradient})
            self.dense_layer_info[layer_index]['new'] = optimizer.update_target(
                self.dense_layer_info[layer_index]['new'], gradient, last_ewa_squared_gradient)
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
            self.dense_layer_info[layer_index]['new'] = optimizer.update_target(
                self.dense_layer_info[layer_index]['new'], gradient, last_momentum, last_ewa_squared_gradient, i)
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
            self.dense_layer_info[layer_index]['new'] = optimizer.update_target(
                self.dense_layer_info[layer_index]['new'], gradient, last_momentum, last_ewa_squared_gradient, i)

    @staticmethod
    def calc_accuracy(y_true, y_prediciton):
        if len(y_true.shape) != 1:
            y_true = y_true.reshape(1, -1)[0]
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
        return np.where(prediction_proba > threshold, 1, 0)

    def _predict_proba(self, X, bias=False):
        x = np.atleast_2d(X)
        if bias:
            x = np.c_[x, np.ones((x.shape[0]))]
        out = x
        for layer_index in self.dense_layer_info.keys():
            o1 = np.dot(out, self.dense_layer_info[layer_index]['new'])
            # out = self.sig_deriv(o1)
            out = self.activation_function(self.dense_layer_info[layer_index]['activation_function'], o1)
        return out

    def show_network(self):
        height = max(self.num_input, max(len(x) for x in self.dense_layer_vertex.values())) - 1
        width = ((self.layer_num + 1) + height) * 1.5
        start_pos_x = width / (self.layer_num + 1)
        if self.bias:
            pos = {
                v_input: (start_pos_x, (height - 0.5) / (self.num_input + 1) * (i + 1))
                for v_input, i in zip(self.input_vertex_list, range(self.num_input + 1))
            }
        else:
            pos = {
                v_input: (start_pos_x, (height - 0.5) / self.num_input * (i + 1))
                for v_input, i in zip(self.input_vertex_list, range(self.num_input))
            }
        pos2 = {
            i: (start_pos_x + key * start_pos_x, (height - 0.5) / len(self.dense_layer_vertex[key]) * (index + 1))
            for key in self.dense_layer_vertex.keys() for index, i in enumerate(self.dense_layer_vertex[key])
        }
        pos.update(pos2)
        plt.title('Basic Neural Network')
        plt.xlim(0, width)
        plt.ylim(0, height)
        nx.draw(
            self.G,
            pos=pos,  # 点的位置
            node_color='red',  # 顶点颜色
            edge_color='black',  # 边的颜色
            with_labels=True,  # 显示顶点标签
            font_size=7 if height > 30 else 13,  # 文字大小
            node_size=100 if height > 30 else 300,  # 顶点大小
            font_color='black'
        )
        plt.show()
