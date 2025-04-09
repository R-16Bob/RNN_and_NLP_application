# 长短期记忆网络LSTM
LSTM是一种Simple RNN的改进，解决RNN的梯度消失和记忆力问题。
<img src="imgs/Pasted image 20250409174953.png">
- LSTM比Simple RNN复杂的多，RNN只有一个参数矩阵，LSTM有四个参数矩阵。
<img src="imgs/Pasted image 20250409175211.png">
- LSTM最重要的设计是传送带，记为C；过去的信息通过传送带输给下一个时刻
- LSTM就是通过传送带来避免梯度消失
# Gates on LSTM
LSTM中有很多Gate，可以有选择地让信息通过。
## Forget gate
<img src="imgs/Pasted image 20250409175426.png">

- 遗忘门由sigmoid和元素乘法两部分组成
- sigmoid的输入是向量a，将其中每个元素映射到(0,1)
- sigmoid计算后得到与a相同维度的f，将c与f按元素相乘得到新的向量
<img src="imgs/Pasted image 20250409175842.png">

- 遗忘门f：与c和h相同形状的向量
	- 0表示不通过
	- 1表示全部通过
<img src="imgs/Pasted image 20250409180221.png">
具体地：遗忘门的输入是上一时刻的h和当前输入x，与矩阵W_f做点乘，并通过sigmoid得到向量f，它的每个元素都在0与1之间。

## Input Gate
<img src="imgs/Pasted image 20250409180438.png">
- 输入门i_t依赖于上一时刻h与当前输入x; 输入门的计算类似遗忘门，有一个参数矩阵W_i做点乘，并通过sigmoid得到i_t

## New value
<img src="imgs/Pasted image 20250409180723.png">
还需要计算New value $\tilde{C}_t$，与遗忘门和输入门类似，也是通过参数矩阵W_c点乘；区别是激活函数是双曲正切tanh，所以元素是在[-1,1]之间

## 更新传送带
<img src="imgs/Pasted image 20250409201300.png">
C_t由两部分相加得到：

- 上一时刻传送带$C_{t-1}$与遗忘门f_t相乘得到的向量
- 新的RNN值$\tilde{C}_t$与输入门i_t相乘得到的向量
这样，通过遗忘门删除了传送带过去的信息，并且通过输入门加入了新的信息

## 输出门
计算传送带的值C_t后，下一步是计算LSTM的输出，也就是状态向量h_t
<img src="imgs/Pasted image 20250409201655.png">

- 首先计算输出门o_t，与遗忘门、输入门的计算方式相同。

## 状态向量
<img src="imgs/Pasted image 20250409201939.png">

- 将输出门o_t与经过tanh的传送带c_t按元素相乘，得到状态向量h_t
- h_t有两个拷贝，一个作为LSTM的输出，另一个传输到下一步

# LSTM的参数数量
有遗忘门、输入门、new value、输出门，这四个模块都有各自的参数矩阵W
- W矩阵的行数是h的维度，列数是h的维度+x的维度
- 所以LSTM的参数量是Simple RNN的4倍

# LSTM的keras实现
- 见notebook
dropout也可以用在LSTM层上，在参数中增加`dropout=0.2`
- 据说dropout并不能提升LSTM的效果，因为overfitting不是由LSTM造成的，而是由embedding层造成的。LSTM层有8000+参数，而embedding层有32000个参数

# 总结
- LSTM使用了传送带，让**过去**的信息很容易传到下一时刻，从而有了更长的记忆。
- LSTM比Simple RNN表现要好