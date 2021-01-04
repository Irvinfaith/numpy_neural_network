@[TOC](python BP神经网络原理 以及 基于numpy的python代码实现)

# 1. 原理
## 1.1 神经网络结构
输入层 + 隐含层 + 输出层
一般来说，每一个隐含层都会需要定义一个激活函数，以此添加非线性的表现。

## 1.2 每层输入结构以及数据流转
输入的数据，假设是10行的样本，有3个特征的数据，即维度(10, 3)的矩阵
### 1.2.1 输入层
那么在神经网络的输入层，就需要有3个神经元，因为有3个特征。所以一次迭代的样本数是1，即batch_size=1的时候，输入的矩阵维度是(1, 3)
### 1.2.2 隐含层
传递到下一层的隐含层（假设隐含层都是全连接层），假设有5个神经元，那么，在这一层之间的权重矩阵，就是维度为(3, 5)的矩阵。所以将输入矩阵(1, 3) 叉乘 权重矩阵(3, 5) 得到 (1, 5) 的输出矩阵，即为这层隐含层的输出了。可以理解为通过线性组合的方式将原始的3个特征增加到了5个特征。
### 1.2.3输出层
一般来说，输出层只有一个神经元，根据激活函数将输出定义为分类/回归任务。
所以在这一层的权重矩阵的维度是(上一层的神经元个数, 1)，也就是(5， 1)

## 1.3 初始权重及权重更新
初始权重是随机设置的，虽说是随机设置，但其实也是有很多方法的，可以按照指定的分布设置随机数，也可以在设置随机权重后除以神经元个数的根号项（Xavier初始化）等等方法。初始了权重之后，就开始进行权重的迭代更新了，具体分为前向传播和反向传播。
### 1.3.1 前向传播
前向传播是很简单的，就是将该层的初始值矩阵乘上权重矩阵，得到的输出再过一次该层的激活函数，得到最终该层的输出，直到最后是输出层的时候，即是最终的输出值，然后根据损失函数计算与目标值之间的损失，根据这个损失进行反向传播

伪代码如下：[查看完整的基于numpy实现的代码](https://github.com/Irvinfaith/numpy_neural_network)
```powershell
# 隐含层
for (每一层 in 隐含层)：
	out = 该层的输入 * 该层的权重
	activation = activation_function(out)
	下一层的输入 = activation
# 输出层
out = 该层的输入 * 该层的权重
activation = activation_function(out)
loss = loss_function(y_true - activation)
```

#### 1.3.1.1 1层隐含层的前向传播

首先定义需用到的变量

- $A$：假设现在的输入特征数据为10条样本，3个特征，即维度为 (10, 3) 的矩阵，设为$A$

- $w_1$： 第一层隐含层有4个神经元，输入层与该层的权重为$w_1$，维度为 (3, 4)
- $out_1$：第一层的原输出
- $f_{activation_1}$：第一层的激活函数
- $activation_1$：第一层的激活输出

- $w_2$：输出层1个神经元，与上一层隐含层的权重为$w2$，维度为 (4，1)
- $out_2$：第二次也就是输出层的原输出
- $f_{activation_2}$：第二层也就是输出层的激活函数
- $activation_2$：输出层的激活输出，也就是该神经网络的正向传播在该轮次的输出结果，该结果直接和目标值进行比对得到损失
- $y_{true}$：真实y值
- $f_{loss}$：损失函数
- $loss$：损失值

首先是正向传播：
$$
out_1 = A\odot w_1
$$

$$
activation_1=f_{activation_1}(out_1)
$$

$$
out_2 = activation_1 \odot w_2
$$

$$
activation_2=f_{activaton_2}(out_2)
$$

$$
loss=f_{loss}(y_{true},activaion_2)
$$

到此得到loss，一轮正向传播就结束了。

### 1.3.2 反向传播

反向传播是神经网络的重点，是更新权重的步骤。
优化方法主要依据梯度下降，很多优化器的演变都是基于梯度下降来的，即根据损失函数梯度的负方向不断逼近，就可以找到最小值。不明白梯度下降的，推荐阅读这篇[梯度下降](https://www.cnblogs.com/pinard/p/5970503.html)的文章，讲解很详细。

#### 1.3.2.1 链式法则
在将反向传播之前，有个很重要的微积分数学定理叫链式法则，也就是在求导的过程中，若目标包裹在多层函数中，需逐一对包裹的函数进行微分。
例如需对如下方程求导：
$$f(x)=sin(x^2+2)，求\frac{df}{dx}$$
可以先写成如下形式：
$f(u)=sin(u), u=x^2+2$
因此根据链式法则：
$$\frac{df}{dx}=\frac{df}{du}*\frac{du}{dx}=cos(u)*2x=2x*cos(x)=2x*cos(x^2+2)$$

#### 1.3.2.2 反向传播

##### 1.3.2.2.1 1层隐含层

![1609294522969](C:\Users\BBD\AppData\Roaming\Typora\typora-user-images\1609294522969.png)

还是用前面的前向传播的例子。

正向传播已经得到了如下的式子，这里再列一遍方便下面公式推导的对照：
$$
out_1 = A\odot w_1\tag{a}
$$

$$
activation_1=f_{activation_1}(out_1)\tag{b}
$$

<div id="activation_1">

$$
out_2 = activation_1 \odot w_2\tag{c}
$$
<div id="out_2">

$$
activation_2=f_{activaton_2}(out_2)\tag{d}
$$

$$
loss=f_{loss}(y_{true},activaion_2)\tag{e}
$$

接下来进行反向传播的推导：

###### $w_2$的更新

首先看对$w_2$的反向传播：
$$
\frac{\partial loss}{\partial w_2}=\frac{\partial loss}{\partial activation_2}\frac{\partial activation_2}{\partial out_2}\frac{\partial out_2}{\partial w_2}
$$




> 为了方便理解和推导，假设这里的所有的激活函数都是sigmoid
> $$
> f(x)=\frac{1}{1+e^{-x}}
> $$
> sigmoid的导数：
> $$
> f'(x)=f(x)(1-f(x)) \tag{1}
> $$
> <div id="sigmoid">
>
> 损失函数是MSE：(乘上$1\over2$为了方便抵消常数项)
> $$
> MSE=\frac{1}{2}\sum (y -\hat y)^2
> $$

1. 先计算第一项$\frac{\partial loss}{\partial activation_2}$

   因为损失函数是MSE，所以：
   $$
   loss=f_{loss}(y_{true},activaion_2)=\frac{1}{2}(y_{true}-activation_2)^2
   $$
   那么：

$$
\frac{\partial loss}{\partial activation_2}=(y_{true}-activation_2)*(-1)=activation_2-y_{true}\tag{2}
$$

2. 中间项$\frac{\partial activation_2}{\partial out_2}$

   因为激活函数是sigmoid，因此：
   $$
   activation_2=f_{activaton_2}(out_2)=\frac{1}{1+e^{-out_2}}​
   $$
   所以，根据[公式（1）](#sigmoid)：

$$
\frac{\partial activation_2}{\partial out_2}=out_2(1-out_2)\tag{3}
$$

3. 最后一项$\frac{\partial out_2}{\partial w_2}$

   因为：$out_2 = activation_1 \odot w_2$

   所以：

$$
\frac{\partial out_2}{\partial w_2}=activaiton_1\tag{4}
$$

最终对$w_2$的反向传播结果：
$$
\frac{\partial loss}{\partial w_2}=(activation_2-y_{true}) out_2(1-out_2)activaiton_1
$$
根据梯度下降的公式，更新$w_2$，其中$\alpha$为学习率（learning rate）：
$$
w_2:=w_2 - \alpha\frac{\partial loss}{\partial w_2}
$$
现在来梳理一下如果应用为矩阵的话，(2) (3) (4) 这每一坨的维度是如何的，该如何进行相乘：

为了阅读起来简洁一些，现在用



- $\Gamma loss$ 表示 $\frac{\partial loss}{\partial activation_2}=activation_2-y_{true}$ ，(m, 1)
  - 输出层**损失函数对激活输出的偏导**，维度：(m, 这一层的神经元个数)，因为只有1个神经元，也就是 (m, 1)

- $\Gamma activation_2$ 表示 $\frac{\partial activation_2}{\partial out_2}=out_2(1-out_2)$， (m, 1)
  - 输出层**激活输出对原输出的偏导**，维度：(m, 这一层的神经元个数) => (m, 1)

- $\Gamma out_2$ 表示 $\frac{\partial out_2}{\partial w_2}=activaiton_1$ ，(m, 4)
  - 输出层**原输出对权重的偏导**，维度： (m, 上一层的神经元个数) => (m, 4)



所以$w_2$的梯度用矩阵表示：
$$
\frac{\partial loss}{\partial w_2}=\Gamma out_2 \odot(\Gamma loss \otimes \Gamma activation_2)
$$

$$
维度：(4, m) \odot((m, 1) \otimes(m, 1))=(4, m) \odot (m, 1)=(4, 1)
$$

得到的维度，刚好就是$w_2$的维度： (上一层神经元个数, 这一层神经元个数) => (4, 1)

所以对$w_2$的更新由矩阵表示：
$$
w_2:=w_2-\alpha*\Gamma out_2 \odot(\Gamma loss \otimes \Gamma activation_2)
$$
其中，$\alpha$为常数，所以可以直接相乘。

到此对$w_2$的权重更新就结束了，这里我们把偏导的前两项包装起来：
$$
\Gamma derive_{out}=\Gamma loss \otimes \Gamma activation_2\tag{5}
$$
$$
维度为：(m, 1)
$$

$\Gamma derive_{out}$会在$w_1$的更新里用上，下面就开始对$w_1$的更新

###### $w_1$的更新

和$w_1$的更新同理，只是长了一小截，先上公式：
$$
\frac{\partial loss}{\partial w_1}=\frac{\partial loss}{\partial activation_2}\frac{\partial activation_2}{\partial out_2}\frac{\partial out_2}{\partial activation_1}\frac{\partial activation_1}{\partial out_1}\frac{\partial out_1}{\partial w_1}
$$
细心的盆友肯定能观察出来，$w_2$更新的前两项和$w_1$更新的前两项是一样的，所以根据式 (5)：
$$
\frac{\partial loss}{\partial w_1}=\Gamma derive_{out}\frac{\partial out_2}{\partial activation_1}\frac{\partial activation_1}{\partial out_1}\frac{\partial out_1}{\partial w_1}
$$
现在看新的第一项偏导：

1. $\frac{\partial out_2}{\partial activation_1}$

   根据 [式(c)](#out_2)：$out_2 = activation_1 \odot w_2$

   因此：

   $$
   \frac{\partial out_2}{\partial activation_1}=w_2
   $$

2. $\frac{\partial activation_1}{\partial out_1}$

   根据 [式(b)](#activation_1)：$activation_1=f_{activation_1}(out_1)$，

   以及[式(1)](#sigmoid)：$f'(x)=f(x)(1-f(x))$

   因此：
   $$
   \frac{\partial activation_1}{\partial out_1}=out_1(1-out_1)
   $$
   
3. $\frac{\partial out_1}{\partial w_1}$

   根据式(a)：$out_1 = A\odot w_1$

   因此：
   $$
   \frac{\partial out_1}{\partial w_1}=A
   $$
   


最终对$w_1$的反向传播结果：
$$
\frac{\partial loss}{\partial w_1}=\Gamma derive_{out}*w_2*out_1(1-out_1)*A
$$

以及根据梯度下降，对$w_1$进行更新：
$$
w_1:=w_1-\alpha\frac{\partial loss}{\partial w_1}
$$
同样的，现在梳理一下每一项的维度是如何的。

方便阅读，同样用以下符号来代替：

- $\Gamma out_2act_1$ 表示 $\frac {\partial out_2}{\partial activation_1}=w_2$， (4, 1)
  - 输出层**原函数对上一层激活输出的偏导**，因为结果是$w_2$， $w_2$ 的维度 (隐含层神经元个数, 输出层神经元个数)，也就是 (4, 1)

- $\Gamma activation_1$ 表示 $\frac{\partial activation_1}{\partial out_1}=out_1(1-out_1)$，(m, 4)
  - 隐含层**激活输出对原输出的偏导**，因为 $out_1$ 是由 $A \odot w_1$ 得到的结果，即 $(m, 3) \odot(3, 4)=(m,4) $ ，所以该结果维度是 (m, 4)

- $\Gamma out_1$ 表示 $\frac{\partial out_1}{\partial w_1}=A$，(m, 3)
  - 隐含层**原输出对输入的偏导**，得到的结果是输入$A$，维度为 (m, 3)

由于：

$\Gamma derive_{out}$维度为 (m, 1)

$w_1$ 的维度为 (3, 4)

所以$w_1$ 的更新用矩阵表示：
$$
\frac{\partial loss}{\partial w_1}=\Gamma out_1^T\odot((\Gamma derive_{out} \odot\Gamma out_2act_1^T)\otimes\Gamma activation_1)
$$

$$
=A^T\odot(\Gamma derive_{out}\odot w_2^T\otimes [out_1(1-out_1)])
$$

用维度表示：
$$
\begin{aligned}(m, 3)^T\odot[(m,1)\odot(4,1)^T\otimes(m,4)]&=(3,m)\odot[(m,1)\odot(1,4)\otimes(m,4)]\\
&=(3,m)\odot[(m,4)\otimes(m,4)]\\
&=(3,m)\odot(m,4)\\
&=(3,4)\end{aligned}
$$



最终得到与 $w_1$ 一致的维度 (3, 4)

所以对 $w_1$更新用矩阵表示：
$$
w_1:=w_1-\alpha*\Gamma out_1^T\odot((\Gamma derive_{out} \odot\Gamma out_2act_1^T)\otimes\Gamma activation_1)
$$
再列一下已计算出的公式：

$\Gamma out_1=\frac{\partial out_1}{\partial w_1}=A$

$\Gamma loss=\frac{\partial loss}{\partial activation_2}=activation_2-y_{true}$

$\Gamma activation_2=\frac{\partial activation_2}{\partial out_2}=out_2(1-out_2)$

$\Gamma derive_{out}=\Gamma loss \otimes \Gamma activation_2=(activation_2-y_{true})\otimes [out_2(1-out_2)]$ 

$\Gamma out_2act_1=\frac {\partial out_2}{\partial activation_1}=w_2$

$\Gamma activation_1=\frac{\partial activation_1}{\partial out_1}=out_1(1-out_1)$

所以：
$$
\begin{aligned}w_1&:=w_1-\alpha*A^T\odot((\Gamma loss \otimes \Gamma activation_2 \odot w_2^T)\otimes [out_1(1-out_1)])\\
&:=w_1-\alpha*A^T\odot[(activation_2-y_{true})\otimes (out_2(1-out_2))]\odot w_2^T\otimes[out_1(1-out_1)]\end{aligned}
$$



到此$w_1$的更新就结束了，也是一轮正向传播+反向传播的全过程。

总结一下需要注意的点：

- 反向传播的时候，是从loss开始逐层往回进行更新的，所以先更新$w_2$再更新$w_1$；但是在更新$w_1$的时候，公式中的$w_2$应该用这轮更新之前的原值。

- 代码实现的时候一定要注意矩阵的维度。

  

##### 1.3.2.2.2 2层隐含层

![1609295935102](C:\Users\BBD\AppData\Roaming\Typora\typora-user-images\1609295935102.png)

为了方便推导规律，假设这里新加的隐含层是从第一层加上的，设置为2个神经元。

2层隐含层的情况，和1层其实大体是差不多的，只是多了一个权重矩阵，现在需要更新的权重矩阵记为：$w_1,w_2,w_3$

这里的$w_2,w_3$和上一节的1层隐含层的更新是基本一样的，分别对应上一节里的$w_1，w_2$
$$
w_2 =>1层隐含层的w_1（输出层前一层的权重矩阵）
$$

$$
w_3=>1层隐含层的w_2(和输出层连接的权重矩阵)
$$

唯一有个地方不同的是在$w_2$更新的时候，更新权重公式会有如下变化：

1. 公式的符号下角标会加1，因为多了一层隐含层，但其实值都是基本一样的
2. 最后一项$\frac{\partial out_2}{\partial w_2}$的结果会有不同，因为在1层隐含层的时候，$w_1$是和输入层连接的权重矩阵，所以$\frac{\partial out_1}{\partial w_1}=A$，但是在2层隐含层的情况，$w_2$是中间的隐含层，既不与输入层连接，也没和输出层连接，所以$\frac{\partial out_2}{\partial w_2}=activation_1$，即为上一层隐含层的激活输出

同样还是先定义下变量：

- $A$：输入特征数据，和1层隐含层的输入一样，为10条样本，3个特征，即维度为 (10, 3) 的矩阵，设为$A$
- $w_1$： 第一层隐含层有2个神经元，输入层与该层的权重为$w_1$，维度为 (3, 2)
- $out_1$：第一层的原输出
- $f_{activation_1}$：第一层的激活函数
- $activation_1$：第一层的激活输出
- $w_2$： 第二层隐含层有4个神经元，上一层与该层的权重为$w_2$，维度为 (2, 4)
- $out_2$：第二层的原输出
- $f_{activation_2}$：第二层的激活函数
- $activation_2$：第二层的激活输出
- $w_3$：输出层1个神经元，与上一层隐含层的神经元为4个，所以维度为 (4，1)
- $out_3$：输出层的原输出
- $f_{activation_3}$：输出层的激活函数
- $activation_3$：输出层的激活输出，也就是该神经网络的正向传播在该轮次的输出结果，该结果直接和目标值进行比对得到损失
- $y_{true}$：真实y值
- $f_{loss}$：损失函数
- $loss$：损失值

还是先来看2层隐含层的正向传播：
$$
out_1 = A\odot w_1
$$

$$
activation_1=f_{activation_1}(out_1)
$$


$$
out_2 = activation_1 \odot w_2
$$

$$
activation_2=f_{activaton_2}(out_2)
$$

$$
out_3=activation_2\odot w_3
$$

$$
activation_3=f_{activation_3}(out_3)
$$

$$
loss=f_{loss}(y_{true},activaion_3)
$$

和1层隐含的正向传播也是类似的，只是因为多了一层隐含层，所以多了红框中的2项

<img src="https://img-blog.csdnimg.cn/20201229181112592.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTU3ODU2Nw==,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:50%;" />

###### $w_3$的更新

和1层隐含层的$w_2$更新是一样的，2层隐含层的$w_2$的更新公式：
$$
\frac{\partial loss}{\partial w_3}=\frac{\partial loss}{\partial activation_3}\frac{\partial activation_3}{\partial out_3}\frac{\partial out_3}{\partial w_3}
$$


$$
\frac{\partial loss}{\partial activation_3}=(y_{true}-activation_3)*(-1)=activation_3-y_{true}
$$

$$
\frac{\partial activation_3}{\partial out_3}=out_3(1-out_3)
$$

$$
\frac{\partial out_3}{\partial w_3}=activaiton_2
$$

所以：
$$
\frac{\partial loss}{\partial w_3}=(activation_3-y_{true}) out_3(1-out_3)activaiton_2
$$


- $\Gamma loss$ 表示 $\frac{\partial loss}{\partial activation_3}=activation_3-y_{true}$，(m, 1)
  - 输出层**损失函数对激活输出的偏导**，维度：(m, 这一层的神经元个数)，因为只有1个神经元，也就是 (m, 1)
- $\Gamma activation_3$ 表示 $\frac{\partial activation_3}{\partial out_3}=out_3(1-out_3)$，(m, 1)
  - 输出层**激活输出对原输出的偏导**，维度：(m, 这一层的神经元个数) => (m, 1)
- $\Gamma out_3$ 表示 $\frac{\partial out_3}{\partial w_3}=activaiton_2$，(m, 4)
  - 输出层**原输出对权重的偏导**，维度： (m, 上一层的神经元个数) => (m, 4)



$w_3$的更新用矩阵表示：
$$
\begin{aligned}w_3:&=w_3-\alpha*\frac{\partial loss}{\partial w_3}\\
&=w_3-\alpha*\Gamma out_3 \odot(\Gamma loss \otimes \Gamma activation_3)\\
&=w_3-\alpha*activaiton_2\odot[(activation_3-y_{true}) \otimes [out_3(1-out_3)]\end{aligned}
$$
这里将$\Gamma loss \otimes \Gamma activation_3$ 记为 $\Gamma derive_{out3}$，其维度为：(m, 1)
$$
\Gamma derive_{out3}=\Gamma loss \otimes \Gamma activation_3
$$


###### $w_2$的更新

和1层隐含层类似，这里就不过多描述了。

需要强调的是，注意找下规律：

- **权重的更新，如果没有和输出层连接，每增加一层，和上一层的更新公式相比，除了共有的部分，更新的公式只有最后3项**
  - 第一项是后一层的原始权重
  - 倒数第二项永远是**该层激活输出对原输出的偏导项**。
  - 若所求的权重**没有和输入层连接**，最后一项的偏导结果是**上一层隐含层的激活输出**；否则最后一项的偏导结果是**输入矩阵**

$w_2$的更新：
$$
\begin{aligned}\frac{\partial loss}{\partial w_2}&=\frac{\partial loss}{\partial activation_3}\frac{\partial activation_3}{\partial out_3}\frac{\partial out_3}{\partial activation_2}\frac{\partial activation_2}{\partial out_2}\frac{\partial out_2}{\partial w_2}\\
&=\Gamma derive_{out3}\frac{\partial out_3}{\partial activation_2}\frac{\partial activation_2}{\partial out_2}\frac{\partial out_2}{\partial w_2}\\
\end{aligned}
$$


1. $\frac{\partial out_3}{\partial activation_2}$

   根据：$out_3 = activation_2 \odot w_3$

   因此：

   $$
   \frac{\partial out_3}{\partial activation_2}=w_3
   $$

2. $\frac{\partial activation_2}{\partial out_2}$ 

   根据  $activation_2=f_{activation_2}(out_2)$，

   以及[式](#sigmoid)：$f'(x)=f(x)(1-f(x))$

   因此：
   $$
   \frac{\partial activation_2}{\partial out_2}=out_2(1-out_2)
   $$
   
3.  $\frac{\partial out_2}{\partial w_2}$

   根据式()：$out_2 = activation_1\odot w_2$

   因此：
   $$
   \frac{\partial out_2}{\partial w_2}=activation_1
   $$

因为这里$w_2$是没有和输入层连接的，所以最后一项的偏导结果是上一层隐含层的激活输出$activation_1$

- $\Gamma out_3act_2$ 表示 $\frac {\partial out_3}{\partial activation_2}=w_3$，(4, 1)
  - 输出层**原函数对上一层激活输出的偏导**，因为结果是 $w_3$， $w_3$ 的维度 (隐含层神经元个数, 输出层神经元个数)，也就是 (4, 1)

- $ \Gamma activation_2$ 表示 $\frac{\partial activation_2}{\partial out_2}=out_2(1-out_2)$，(m, 4)
  - 隐含层**激活输出对原输出的偏导**，因为 $out_2$ 是由 $activation_1 \odot w_2$ 得到的结果，即 $(m, 2) \odot(2, 4)=(m,4) $ ，所以该结果维度是 (m, 4)

- $\Gamma out_2$ 表示 $\frac{\partial out_2}{\partial w_2}=activation_1$ ，(m, 2)
  - 隐含层**原输出对输入的偏导**，得到的结果是上一层的激活输出，维度为 (m, 2)

由于：

$\Gamma derive_{out_3}$ 维度为 (m, 1)

$w_2$ 的维度为 (2, 4)

所以$w_2$的更新用矩阵表示（和1层隐含层的$w_1$ 更新基本一致）：
$$
\frac{\partial loss}{\partial w_2}=\Gamma out_2^T\odot((\Gamma derive_{out_3} \odot\Gamma out_3act_2^T)\otimes\Gamma activation_2)
$$

$$
=activation_1^T\odot(\Gamma derive_{out_3}\odot w_3^T\otimes [out_2(1-out_2)])
$$

用维度表示：
$$
\begin{aligned}(m, 2)^T\odot[(m,1)\odot(4,1)^T\otimes(m,4)]&=(2,m)\odot[(m,1)\odot(1,4)\otimes(m,4)]\\
&=(2,m)\odot[(m,4)\otimes(m,4)]\\
&=(2,m)\odot(m,4)\\
&=(2,4)\end{aligned}
$$



最终得到与 $w_2$一致的维度 (2, 4)
$$
用\Gamma derive_{out_2}记为：\Gamma derive_{out3}\frac{\partial out_3}{\partial activation_2}\frac{\partial activation_2}{\partial out_2}
$$
$w_2$的更新就完成了



###### $w_1$的更新

以此类推，就不详细写公式了，直接按规律来。

重新再列一下由后面更新计算过的层传递过来的中间值：

$\Gamma derive_{out3}=\frac{\partial loss}{\partial activation_3}\frac{\partial activation_3}{\partial out_3}$

$\Gamma derive_{out_2}=\Gamma derive_{out3}\frac{\partial out_3}{\partial activation_2}\frac{\partial activation_2}{\partial out_2}$

因此：
$$
\begin{aligned}\frac{\partial loss}{\partial w_1}&=(\frac{\partial loss}{\partial activation_3}\frac{\partial activation_3}{\partial out_3}\frac{\partial out_3}{\partial activation_2})\frac{\partial activation_2}{\partial out_2}\frac{\partial out_2}{\partial activation_1}\frac{\partial activation_1}{\partial out_1}\frac{\partial out_1}{\partial w_1}\\
&=(\Gamma derive_{out3}\frac{\partial out_3}{\partial activation_2}\frac{\partial activation_2}{\partial out_2})\frac{\partial out_2}{\partial activation_1}\frac{\partial activation_1}{\partial out_1}\frac{\partial out_1}{\partial w_1}\\
&=\Gamma derive_{out_2}\frac{\partial out_2}{\partial activation_1}\frac{\partial activation_1}{\partial out_1}\frac{\partial out_1}{\partial w_1}\\
\end{aligned}
$$
现在来看剩下的这3项，是不是就可以根据前面已经得到的规律闭着眼睛写出结果了？

1. $\frac{\partial out_2}{\partial activation_1}$

$$
\frac{\partial out_2}{\partial activation_1}=w_2
$$

2. $\frac{\partial activation_1}{\partial out_1}$
   $$
   \frac{\partial activation_1}{\partial out_1}=out_1(1-out_1)
   $$

3. $\frac{\partial out_1}{\partial w_1}$

$$
\frac{\partial out_1}{\partial w_1}=A
$$

还记得前面讲的么，因为这一层已经和输入层连接了，所以最后一项的偏导结果就是输入矩阵，而不是前一层的激活输出了，这里都没前一层了。

维度也是一样的，这里就不列出来了。

#### 1.3.2.3 总结

反向传播的时候，从后逐层往前进行计算更新，由于中间会有很多重复的计算，并且具有差异的项也有一定的规律，具体规律如下：

1. 最后一个隐含层和输出层之间的权重是第一个更新的权重矩阵，将该层的矩阵设为$w_n$，这个矩阵的更新因为是第一步，只能老老实实进行计算，需注意的是，这一层更新权重公式的3项偏导项，需将前两项的结果传递到后面的更新中，将这两项设为$\Gamma derive_n$，

   ```python
   # 计算误差
   error = tmp_value[list(tmp_value.keys())[-1]]['activation'] - y
   # this_out_deriv = self.sig_deriv(tmp_value[layer_index]['out'], True)
   this_out_deriv = self.activation_function(self.dense_layer_info[layer_index['activation_function'],tmp_value[layer_index]['out'],True)
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
   ```

   

2. 再往前传递若干层，假设这若干层都是没有和输入层连接的，那么每计算更新一层权重，新增的3个偏导项，都需将前两项包装，与前面传递过来的$\Gamma derive $ 通过一定方式组合相乘起来，传递到下一步更新中。

   如何组合呢：

   这里先上一段代码：

   ```
   # 接收前面层传递的链式偏导并继续往下层传递, shape: (m, next_n)
   derive = np.multiply(last_derive.dot(last_ori_weight.T), this_out_deriv)
   ```

   其中：

   - last_derive：是上一层传递过来的$\Gamma derive $

   - last_ori_weight：是该层需包装的前两项中的第一项

     比如上面的更新$w_1$的时候：
     $$
     \frac{\partial out_2}{\partial activation_{1}}=w_2
     $$
     也就是上一层的**原始权重**

   - this_out_derive：是该层需包装的前两项中的第二项

     比如上面更新$w_1$里的：
     $$
     \frac{\partial activation_1}{\partial out_1}=out_1(1-out_1)
     $$
     也就是这一层激活输出对原输出的偏导，也就是激活函数的导数

   并且乘在一起之后，得到的$\Gamma derive $的维度为 **(m, 该层的神经元个数)**

   以及这一层新增的最后一项偏导的结果就是上一层的激活输出，维度为**(m, 上一层的神经元个数)**

   ```
   gradient = last_activation.T.dot(derive)
   ```

   $$
   (m, 上一层的神经元个数)^T\odot \Gamma (m, 该层的神经元个数)=(上一层神经元个数, 该层神经元个数)
   $$

   这样得到最后这层的梯度维度刚好就是这层权重矩阵的维度了。

   ```python
   # last_derive: 前一层传递的偏导，shape: (m, next_n下一层神经元个数)
   last_derive = tmp_value[layer_index + 1]['derive']
   # last_ori_weight: 上一层的权重矩阵，shape: (this_n, next_n) (当前层神经元个数, 下层神经元个数)
   last_ori_weight = self.dense_layer_info[layer_index + 1]['ori']
   # this_out_deriv: 该层输出层的偏导，shape: (m, this_n)
   # this_out_deriv = self.sig_deriv(tmp_value[layer_index]['out'], True)
   this_out_deriv = self.activation_function(self.dense_layer_info[layer_index]['activation_function'],tmp_value[layer_index]['out'],True)
   # 接收前面层传递的链式偏导并继续往下层传递, shape: (m, next_n)
   derive = np.multiply(last_derive.dot(last_ori_weight.T), this_out_deriv)
   tmp_value[layer_index].update({'derive': derive})
   # last_activation: 上一层的激活输出，shape: (m, this_n)
   last_activation = tmp_value[layer_index - 1]['activation']
   
   # Optimizer: 计算梯度gradient，shape：(this_n, next_n)
   gradient = last_activation.T.dot(derive)
   # 执行优化器，更新参数
   self.cal_optimizer(optimizer, gradient, layer_index, i, tmp_value)
   ```

   

3. 如果权重矩阵和输入层连接了，也就意味该轮的反向传播到了最后一步了，这层新增的3个偏导项，只有最后一项会与前面所寻的规律有所不同，最后一项的偏导结果是输入矩阵，而不是像之前的 前一层的激活输出。

   ```
   # x为输入层输入的特征矩阵
   gradient = x.T.dot(derive)
   ```

   x的维度为(m, n)，n是特征数，也就是输入层的神经元个数

   所以：
   $$
   (m, 输入层神经元个数)^T\odot \Gamma (m, 该层的神经元个数)=(输入层神经元个数, 该层神经元个数)
   $$
   

   ```python
   # last_derive: 前一层传递的偏导，shape: (m, next_n下一层神经元个数)
   last_derive = tmp_value[layer_index + 1]['derive']
   # last_ori_weight: 上一层的权重矩阵，shape: (this_n, next_n) (当前层神经元个数, 下层神经元个数)
   last_ori_weight = self.dense_layer_info[layer_index + 1]['ori']
   # this_out_deriv: 该层输出层的偏导，shape: (m, this_n)
   # this_out_deriv = self.sig_deriv(tmp_value[layer_index]['out'], True)
   this_out_deriv =self.activation_function(self.dense_layer_info[layer_index['activation_function'],tmp_value[layer_index]['out'],True)
   # 接收前面层传递的链式偏导并继续往下层传递, shape: (m, next_n)
   derive = np.multiply(last_derive.dot(last_ori_weight.T), this_out_deriv)
   tmp_value[layer_index].update({'derive': derive})
   # 计算梯度gradient：
   gradient = x.T.dot(derive)
   # 执行优化器，更新参数
   self.cal_optimizer(optimizer, gradient, layer_index, i, tmp_value)
   ```



到此神经网络的正向传播和反向传播的推导以及规律就总结完了。

[查看完整的基于numpy实现的代码](https://github.com/Irvinfaith/numpy_neural_network)



# 2. 损失函数

## 2.1 MSE



```python
def mse(true_y, prediction_y, derive=False):
    total_loss = 1 / 2 * np.sum((true_y - prediction_y) ** 2)
    if derive:
        return prediction_y - true_y
    return total_loss
```



## 2.2 MAE

```python
def mae(true_y, prediction_y, derive=False):
    if derive:
        return 1 / true_y.shape[0] * np.abs(prediction_y - true_y)
    else:
        return 1 / true_y.shape[0] * np.sum(np.abs(true_y - prediction_y))
```





# 3. 优化器

本篇的重点在于如何用numpy实现这些优化器。优化器的原理和优缺点很多文章都有介绍，[推荐这篇](https://www.cnblogs.com/guoyaohua/p/8542554.html)，讲解还是很详细了。

## 3.1 SGD

SGD (Stochastic Gradient Descent)，随机梯度下降

在神经网络中，如果是SGD的优化方法，意味着在一个epoch中，每一轮正向传播+反向传播的过程中只有一条样本。

若总的样本数是m，一个epoch里会进行m轮正向传播+反向传播，每一轮只使用1条样本，也就是(1, n)的矩阵进入到网络中，并且是随机的一条样本。

所以若有$n_{epoch}$次epoch，网络总共会反复更新 $n_{epoch}*m$次

```python
elif optimizer.name == "SGD" and batch_size is None:
    _shuffle_index = np.random.permutation(x.shape[0])
    for _, data in enumerate(zip(x, y), start=1):
        self._fit_one_round(data[0][_shuffle_index], data[1][_shuffle_index], optimizer, _)
```



## 3.2 BSGD

BSGD (mini-Batch Stochastic Gradient Descent)，小批量随机梯度下降

基于SGD的改进，每一轮不再是只有1条样本，而是设定的`batch_size`条样本进入到一轮的正向传播+反向传播中

```python
elif batch_size:
    m = x.shape[0]
    batch_round = int(m // batch_size)
    # 轮次从1开始，防止计算Adam的时候初始轮次为0导致除数为0
    for _ in range(1, batch_round + 1):
        index = np.random.choice(np.arange(m), batch_size, replace=False)
        self._fit_one_round(x[index, :], y[index, :], optimizer, _)
```

可以看到总共的迭代次数减少至$n_{epoch}*\frac{m}{batch\_size}$ 



## 3.3 Momentum
[On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)

计算了基于梯度的移动指数平均（一阶梯度）

```python
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
```



## 3.4 AdaGrad

[Adaptive Subgradient Methods for
Online Learning and Stochastic Optimization](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
AdaGrad 是在普通的梯度下降基础上，计算了一个基于累积梯度平方和的自适应比重（二阶梯度），作为学习率的自适应参数。适用于稀疏的数据集

```python
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
```



## 3.5 AdaDelta

[Adadelta - an adaptive learning rate method](https://arxiv.org/pdf/1212.5701.pdf)
AdaDelta 是在AdaGrad基础上做了很多改进。论文中提出AdaGrad会存储所有的累加梯度评分和，这样在计算的时候并不高效，所以采用了计算了一个基于梯度平方和的指数移动平均的自适应比重，作为学习率的自适应参数。

![1609382958716](C:\Users\BBD\AppData\Roaming\Typora\typora-user-images\1609382958716.png)

并且讲学习率也采用了同样的方法，彻底摒弃了学习率的这个参数，采用完全自适应的方法进行学习

![1609383151244](C:\Users\BBD\AppData\Roaming\Typora\typora-user-images\1609383151244.png)

```python

def calc_ewa_squared_value(self, last_ewa_squared_value, this_value):
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
    ewa_squared_gradient = self.calc_ewa_squared_value(last_ewa_squared_gradient, gradient)
    rms_gradient = self.calc_rms_value(ewa_squared_gradient)
    last_rms_delta_x = self.calc_rms_value(last_ewa_squared_delta_x)
    delta_x = self.calc_delta_x(last_rms_delta_x, rms_gradient, gradient)
    return target + self.alpha * delta_x
```





## 3.4 RMSprop

RMSprop可以说是简化版的AdaDelta，没有采用自动调整的学习率，其他都是一样的。

```python
def calc_rprop(self, gradient, last_rprop):
    """
        计算梯度平方的移动平均

        Args:
            gradient: 上一层传递的梯度
            last_rprop: 上一轮的rprop

        Returns:
            np.array: 本轮的rprop
        """
    rprop = self.gamma * last_rprop + (1 - self.gamma) * np.power(gradient, 2)
    return rprop

def update_target(self, target, gradient, last_rprop):
    """更新目标权重

        Args:
            target: 待更新的目标权重矩阵
            gradient: 上一层传递的梯度
            last_rprop: 本轮的rprop

        Returns:
            np.array: 更新后的目标权重矩阵
        """
    # gamma: 梯度平方移动平均的衰减率; epsilon: 模糊因子，防止除数为零，通常为很小的数
    rprop = self.calc_rprop(gradient, last_rprop)
    delta = self.alpha * gradient / (np.sqrt(rprop) + self.epsilon)
    return target - delta
```



## 3.5 Adam
[ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION](https://arxiv.org/pdf/1412.6980v8.pdf)

Adam是目前用到的较多的优化器了，集成了Momentum和RMSProp/AdaDelta的方法，对梯度既使用了动量，也增加了指数平均的梯度平方和，最终还对两项值进行了优化，解决训练初始时，初始值为0，导致参数逼近会趋近于0的情况，所以引入了轮次的参数。

```python
def calc_correction_momentum(self, gradient, last_momentum, round_num):
    """
        计算修正的动量，减少训练初期因为初始值为0所以会偏向至0的影响

        Args:
            gradient: 上一层的梯度
            last_momentum: 上一轮的动量
            round_num: 迭代训练的轮次号

        Returns:

        """
    momentum = self.calc_momentum(gradient, last_momentum)
    return momentum / (1 - np.power(self.beta_1, round_num))

def calc_correction_rprop(self, gradient, last_rprop, round_num):
    rprop = self.calc_rprop(gradient, last_rprop)
    return rprop / (1 - np.power(self.beta_2, round_num))

def update_target(self, target, gradient, last_momentum, last_rprop, round_num):
    # https://arxiv.org/pdf/1412.6980v8.pdf
    correction_momentum = self.calc_correction_momentum(gradient, last_momentum, round_num)
    correction_rprop = self.calc_correction_rprop(gradient, last_rprop, round_num)
    return target - self.alpha * correction_momentum / (np.sqrt(correction_rprop) + self.epsilon)
```



## 3.6 AdaMax

AdaMax是Adam作者在论文中提出的一种变体方法，将指数移动平均的梯度平方和取上一轮和这一轮所计算的最大值，作为这一轮迭代更新的量。并且不需要计算修正的指数移动平均梯度平方和。

```python
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

```




# 4. 激活函数
## 4.1 sigmoid

```python
def sigmoid(x, derive=False):
    if derive:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))
```



## 4.2 softmax
## 4.3 relu

```python
def relu(x, derive=False):
    if derive:
        return np.where(x > 0, 1, 0)
    return np.where(x > 0, x, 0)
```



## 4.4 tanh

```python
def tanh(x, derive=False):
    if derive:
        return 1 - np.power(tanh(x), 2)
    return 2 * sigmoid(np.multiply(2, x)) - 1
```



# 5. python代码实现

比较多，就不附在这里了，感兴趣的可以去我的github上看看，可以随意搭建神经网络，支持上述所讲的所有优化器、激活函数，并加入了dropout，后续还会逐步更新卷积等复杂网络的代码和推导文章。

下面是numpy神经网络样例代码，[查看完整的numpy实现代码，github地址：https://github.com/Irvinfaith/numpy_neural_network](https://github.com/Irvinfaith/numpy_neural_network)



```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 引入训练数据
data_loader = load_breast_cancer()
data = data_loader['data']
# 归一化
mms = MinMaxScaler()
data = mms.fit_transform(data)
# 拆分训练测试集
X_train, X_test, y_train, y_test = train_test_split(data, data_loader['target'], test_size=0.3, random_state=101)

# 搭建神经网络，输入为30个特征，所以是30个神经元的输入，False表示不加入bias项
nn = NeuralNetwork(30, False)
# 添加全连接层，64个神经元，激活函数为sigmoid
nn.add_dense_layer(64, "sigmoid")
# 添加全连接层，32个神经元，激活函数为relu
nn.add_dense_layer(32, "relu")
# 添加dropout层，留存率0.8
nn.add_dropout_layer(0.8)
# 添加输出层，默认1个神经元
nn.add_output_layer()
# 支持多种优化器
# optimizer = AdaGrad(alpha=0.01)
# optimizer = AdaDelta(alpha=1, beta=0.95)
# optimizer = RMSProp(alpha=0.001, beta=0.9)
optimizer = Adam(alpha=0.05, beta_1=0.9, beta_2=0.99)
# optimizer = Adamax(alpha=0.05, beta_1=0.9, beta_2=0.99)
# 带momentum的SGD
# optimizer = SGD(alpha=0.05, beta=0.99)
# optimizer = SGD(alpha=0.05)
# 开始训练，设置batch_size和优化器，默认损失函数是mse
nn.fit(X_train, y_train, epoch=100, batch_size=64, optimizer=optimizer)
# 预测标签
prediction_y = nn.predict(X_test)
# 预测概率
prediction_proba = nn.predict_proba(X_test)
# 计算准确率
nn.calc_accuracy(y_test, prediction_y)
```

