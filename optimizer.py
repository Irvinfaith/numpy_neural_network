# -*- coding: utf-8 -*-
"""
Created on 2020/12/22 13:04

@author: Irvinfaith

@email: Irvinfaith@hotmail.com
"""
import numpy as np


class Optimizer:
    def __init__(self):
        pass

    def update_target(self, *args, **kwargs):
        pass


class SGD(Optimizer):
    def __init__(self, alpha=0.01, beta=None):
        """
        Args:
            alpha: 学习率
            beta: 动量加速率
        """
        self.alpha = alpha
        self.beta = beta
        super().__init__()

    @property
    def name(self):
        return "SGD"

    def calc_momentum(self, gradient, last_momentum):
        """计算动量，根据上一轮的梯度更新，若梯度方向没改变，梯度持续增加，加快下降；
        若梯度方向改变，则会加上一个相反符号的梯度，即会减少梯度下降的速度，减轻震荡。

        Args:
            gradient: 上一层传递的梯度
            last_momentum: 上一轮的动量

        Returns:

        """
        # http://www.cs.toronto.edu/~fritz/absps/momentum.pdf (1)
        # return self.beta * last_momentum + self.alpha * gradient
        return self.beta * last_momentum + (1 - self.beta) * gradient

    def update_target(self, target, gradient, this_momentum=None):
        """更新目标权重

        Args:
            target: 待更新的目标权重矩阵
            gradient: 上一层传递的梯度
            this_momentum: 本轮的动量

        Returns:
            np.array: 更新后的目标权重矩阵
        """
        if self.beta:
            return target - self.alpha * this_momentum
        return target - self.alpha * gradient


class BGD(SGD):
    def __init__(self, alpha=0.01):
        super().__init__(alpha)

    @property
    def name(self):
        return "BGD"


class AdaGrad(Optimizer):
    def __init__(self, alpha=0.01, epsilon=1e-8):
        """

        Args:
            alpha: 学习率
            epsilon: 模糊因子，防止除数为零
        """
        self.alpha = alpha
        self.epsilon = epsilon
        super().__init__()

    @property
    def name(self):
        return "AdaGrad"

    def update_target(self, target, gradient, total_squared_gradient_sum):
        """更新目标权重

        Args:
            target: 待更新的目标权重矩阵
            gradient: 上一层传递的梯度
            total_squared_gradient_sum: 累积的历史梯度平方和

        Returns:
            np.array: 更新后的目标权重矩阵
        """
        # Optimizer: 计算累积梯度平方和 -> AdaGrad
        ada_grad = gradient / (np.sqrt(total_squared_gradient_sum) + self.epsilon)
        return target - self.alpha * ada_grad


class AdaDelta(Optimizer):
    def __init__(self, alpha=1, beta=0.95, epsilon=1e-8):
        """

        Args:
            alpha: 学习率，论文中是不需要学习率的，这里还是保留，默认为1
            beta: 累积梯度平方衰减率
            epsilon: 模糊因子，防止除数为零
        """
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        super().__init__()

    @property
    def name(self):
        return "AdaDelta"

    def calc_ewa_squared_value(self, this_value, last_ewa_squared_value):
        # https://arxiv.org/pdf/1212.5701.pdf (8)
        return self.beta * last_ewa_squared_value + (1 - self.beta) * np.power(this_value, 2)

    def calc_rms_value(self, ewa_squared_value):
        # https://arxiv.org/pdf/1212.5701.pdf (9)
        return np.sqrt(ewa_squared_value + self.epsilon)

    def calc_delta_x(self, last_rms_delta_x, rms_gradient, gradient):
        # https://arxiv.org/pdf/1212.5701.pdf (10)
        return - last_rms_delta_x / rms_gradient * gradient

    def update_target(self, target, gradient, last_ewa_squared_gradient, last_ewa_squared_delta_x):
        """更新目标权重

        Args:
            target: 待更新的目标权重矩阵
            gradient: 上一层传递的梯度
            last_ewa_squared_gradient: 根据上一轮的移动平均梯度平方和计算的自适应的梯度
            last_ewa_squared_delta_x: 根据上一轮的移动平均自适应率平方和计算的自适应率

        Returns:
            np.array: 更新后的目标权重矩阵
        """
        # Optimizer: 计算移动累积梯度平方和
        ewa_squared_gradient = self.calc_ewa_squared_value(gradient, last_ewa_squared_gradient)
        rms_gradient = self.calc_rms_value(ewa_squared_gradient)
        last_rms_delta_x = self.calc_rms_value(last_ewa_squared_delta_x)
        delta_x = self.calc_delta_x(last_rms_delta_x, rms_gradient, gradient)
        return target + self.alpha * delta_x


class RMSProp(AdaDelta):
    # http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    def __init__(self, alpha=0.001, beta=0.9, epsilon=1e-8):
        """

        Args:
            alpha: 学习率
            beta: 梯度平方移动平均的衰减率
            epsilon: 模糊因子，防止除数为零
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    @property
    def name(self):
        return "RMSProp"

    def update_target(self, target, gradient, last_ewa_squared_gradient):
        """更新目标权重

        Args:
            target: 待更新的目标权重矩阵
            gradient: 上一层传递的梯度
            last_ewa_squared_gradient: 上一轮的指数平均梯度平方和

        Returns:
            np.array: 更新后的目标权重矩阵
        """
        # gamma: 梯度平方移动平均的衰减率; epsilon: 模糊因子，防止除数为零，通常为很小的数
        ewa_squared_gradient = self.calc_ewa_squared_value(gradient, last_ewa_squared_gradient)
        delta = self.alpha * gradient / (np.sqrt(ewa_squared_gradient) + self.epsilon)
        return target - delta


class Adam(SGD, RMSProp):
    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-8):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        SGD.__init__(self, alpha, beta_1)
        RMSProp.__init__(self, alpha, beta_2, epsilon)

    @property
    def name(self):
        return "Adam"

    def calc_correction_momentum(self, gradient, last_momentum, round_num):
        """
        计算修正的动量，减少训练初期因为初始值为0所以会偏向至0的影响

        Args:
            gradient: 上一层的梯度
            last_momentum: 上一轮的动量
            round_num: 迭代训练的轮次数

        Returns:

        """
        momentum = self.calc_momentum(gradient, last_momentum)
        return momentum / (1 - np.power(self.beta_1, round_num))

    def calc_correction_rprop(self, gradient, last_rprop, round_num):
        """

        Args:
            gradient: 本轮的梯度
            last_rprop: 上一轮的指数移动平均梯度平方和
            round_num: 迭代训练的轮次数

        Returns:

        """
        rprop = self.calc_ewa_squared_value(gradient, last_rprop)
        return rprop / (1 - np.power(self.beta_2, round_num))

    def update_target(self, target, gradient, last_momentum, last_rprop, round_num):
        # https://arxiv.org/pdf/1412.6980v8.pdf
        correction_momentum = self.calc_correction_momentum(gradient, last_momentum, round_num)
        correction_rprop = self.calc_correction_rprop(gradient, last_rprop, round_num)
        return target - self.alpha * correction_momentum / (np.sqrt(correction_rprop) + self.epsilon)


class Adamax(Adam):
    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-8):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        super().__init__()

    @property
    def name(self):
        return "Adamax"

    def update_target(self, target, gradient, last_momentum, last_ewa_squared_gradient, round_num):
        """取轮次中最大的梯度平方移动平均数

        Args:
            target:
            gradient:
            last_momentum: 上一轮的动量
            last_ewa_squared_gradient: 上一轮的指数移动平均梯度平方和
            round_num: 迭代训练的轮次数

        Returns:

        """
        # https://arxiv.org/pdf/1412.6980v8.pdf # 7.1
        correction_momentum = self.calc_correction_momentum(gradient, last_momentum, round_num)
        max_ewa_squared_gradient = max(np.linalg.norm(last_ewa_squared_gradient), np.linalg.norm(gradient))
        return target - self.alpha / (1 - np.power(self.beta_2, round_num)) * correction_momentum / (np.sqrt(max_ewa_squared_gradient) + self.epsilon)
