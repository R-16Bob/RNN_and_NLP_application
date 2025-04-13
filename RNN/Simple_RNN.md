[keras文档](https://keras.io/api/)
# How to model sequential data?（怎样对时序数据建模？）
## One to one
 一个输入，一个输出，全连接和CNN都是one to one模型；One to one适合图片问题，但不适合文本问题。
### Limitations of FC Net and ConvNets
- 整体处理一段话，但文本有逐步积累的过程
- 输入长度固定：图片是这样，但文本不是
- 输出长度固定：预测概率是这样，但如果要翻译呢？
## Many to one (many)
对于时序数据（例如文本、语言、时间序列），更好的模型是RNN（循环神经网络）
# Recurrent Neural Networks（RNNs）
<img src="Pasted image 20250331230040.png">
- 与人的阅读习惯类似，一次看一个词，逐渐在大脑里积累信息；RNN每次读一个wrod（通过word embedding为x)，用状态向量h_t积累阅读信息。
- 也就是说，h0包含第一个词the的信息，h1包含前两个词the cat的信息，以此类推...
- 可以将ht看做从这句话抽取的特征向量
- 更新状态向量通过参数矩阵A，参数A只有一个。随机初始化，然后用训练数据学习A。

# Simple RNN
## 状态向量h如何得到?
<img src="Pasted image 20250331231146.png">

- 将前一时刻的状态$h_{t-1}$和当前时刻输入$x_t$做concatenation（连接）；用RNN的参数A乘这个矩阵，在将激活函数tanh用到结果矩阵的每一个元素上
>tanh（hyperbolic tangent，双曲正切）：R->[-1,1] 
<img src="Pasted image 20250331231302.png">

### 为什么需要双曲正切激活函数？
假设输入的词向量x=0.
$$h_{100}=Ah_{99}=A^2h_{98}=\cdots=A^{100}h_{0}$$
- 假设矩阵A的最大特征值略小于1，设为0.9；当A到100次幂时，A^100的特征值将是0.9^100，也就是接近0的数；由于A^100的所有特征值都非常小，矩阵本身将是一个接近0的矩阵。
- 假设矩阵A的最大特征值略大于1，设为1.2；它的100次方将特别大，h_100也将非常大。
使用tanh可以让A不会全0或者太大。

RNN参数数量=A的参数数量=shape(h)x\[shape(h)+shape(x)\]

# Simpe RNN for Movie Review Analysis
可以使用交叉验证选择最优的维度。（比较选择不同维度时的准确率。
>Cross Evaluation（交叉验证）是一种用于评估机器学习模型性能的技术，通过将数据集划分为多个子集（通常称为“折”），然后多次训练和验证模型，以评估其在不同数据子集上的表现。这种方法可以有效减少模型对特定训练集和验证集的依赖，从而更准确地评估模型的泛化能力。
<img src="Pasted image 20250331233453.png">

只需要输出h_t，因为它包含了整句话的信息。

如果让RNN返回所有的状态向量，则只需要加一个Flatten：
<img src="Pasted image 20250401000402.png">

# Simple RNN的缺陷
- it is good at short-term dependence： 比如the clound on the (sky)
- 它的缺点是不擅长long-term dependence: 因为h_100与x_1几乎无关，梯度为0，这显然不合理