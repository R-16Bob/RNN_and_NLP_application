>下文中的x_:i实际上就是x_1的意思，也就是第一个x值。
# Attention without RNN
Transformer完全基于Attention。
>Attention最初是用在RNN上的，本节把RNN去掉，只保留Attention。

原论文：
Vaswani, Ashish, et al. "[Attention is all you Need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) ." _Advances in neural information processing systems_ 30 (2017).

## Transformer 模型
- Transformer是一个Seq2Seq模型。它有一个Encoder和一个Decoder，很适合机器翻译。
- Transformer不是RNN。
- Transformer没有循环的结构，只有Attention和全连接层。
- Transformer的使用效果惊人，完爆最好的RNNs。
机器翻译已经没有人用RNN了，业界都用Transformer+BERT

### 如何搭一个只基于Attention的神经网络？
<img src="imgs/Pasted image 20250414173225.png">

#### 回顾SimpleRNN+Attention
- Encoder：将输入的信息(token)压缩到状态向量h中，最后一个状态h_m包含所有信息。
- Decoder是一个文本生成器，根据状态s生成单词；并将生成的单词作为下一个输入 $$x^{\prime}$$ ；还需要计算contex vector c，每计算一个状态s就需要计算一个c；c是通过注意力，也就是比较当前状态s与Encoder的每一个状态h的相关性计算权重：

$$\alpha_{i,j}=\mathrm{align}(\mathbf{h}_i,\mathbf{s}_j)$$
计算

$$k_{:i}=\mathbf{W}_Kh_i$$ 
(将所有k组成矩阵K)and 

$$q_{:i}=\mathbf{W}_Qs_j$$

计算权重：

$$\alpha_{:j}=\mathrm{Softmax}(\mathbf{K}^Tq_{:j})\in\mathbb{R}^m$$
<img src="imgs/Pasted image 20250416154125.png">
上面对Encoder状态h和Decoder状态s做线性变化得到的向量分别是k和q，它们被称为Key和Query：
- **Query:** $$q_{:i}=\mathbf{W}_Qs_j$$
(To match others)
- **Key:** $$k_{:i}=\mathbf{W}_Kh_i$$ 
(To be matched)
用一个query向量与m个Key匹配，是通过与矩阵K相差，得到的alpha表示query与对应K的匹配程度。

Transformer中还有第三个参数矩阵W_V
- **Value:** $$v_{:i}=\mathbf{W}_Vh_j.$$ 
(To be weighted averaged.)

Value的作用是用来和权重加权平均得到上下文向量（不再使用未经处理的h）
- Context vector:

$$c_j=\alpha_{1j}v_{:1}+\cdots+\alpha_{mj}v_{:m}.$$

这种通过Query，Key， Value计算权重和上下文向量的方式就是Transformer用的。

### 如何去掉RNN只保留Attention？
Transformer是由Attention层与Self-Attention层组成的

#### Attention Layer
- 研究Seq2Seq模型（Encoder+Decoder)
- Encoder输入：向量$$\mathbf{x}_1,\mathbf{x}_2, \cdots, \mathbf{x}_m$$

- Decoder输入：向量 $$\mathbf{x}_1^{\prime},\mathbf{x}_2^{\prime}, \cdots, \mathbf{x}_m^{\prime}$$

首先，基于Encoder的输入计算Keys与Values向量：
- **Key:** $$k_{:i}=\mathbf{W}_K\mathbf{x}_i.$$
- **Value**: $$v_{:i}=\mathbf{W}_V\mathbf{x}_i$$
基于Decoderd的输入做线性变换，得到Queries向量：
- **Query:** $$1_{:j}=\mathbf{W}_Q\mathbf{x}^{\prime}_j$$
计算权重Weights：

$$\alpha_{:1}=\mathrm{Softmax}(\mathbf{K}^Tq_{:1})\in\mathbb{R}^m$$

<img src="imgs/Pasted image 20250416161318.png">

计算Context Vector:

$$c_{:1}=\alpha_{11}v_{:1}+\cdots+\alpha_{m1}v_{:m}=V\alpha_{:1}.$$

也就是dim（V）x m的矩阵乘 mx1的权重向量alpha_:1得到dim(v) x 1的Context Vector，实际做的就是对V加权平均
<img src="imgs/Pasted image 20250416161401.png">
- attention layer的输出: $$C=[c_{:1},\cdots,c_{:t}].$$
- Here, $$c_{:j}=V\cdot\mathrm{Softmax}(K^Tq_{:j}).$$
- Thus，c_:j是依赖于 $$x^{\prime}_j$$ 和 $$[x_1,x_2,\cdots,x_m]$$ 的函数。
<img src="imgs/Pasted image 20250416162919.png">

以英语-德语翻译为例，c_:2 能够看到Encoder的所有输入和Decoder的当前输入；将它接入softmax分类器，就可以得到下一个词的预测。

>也就是说，W_V其实充当了RNN得到隐藏状态h的作用，value就相当于RNN中的h。
>总结：RNN与Attention做机器翻译非常类似，RNN用h做特征向量，而Attention用contex vector c作为特征向量。由于输入输出长度相同，可以直接将Attention layer代替RNN构建一个Seq2Seq模型。

使用Attention layer的好处是可以避免遗忘，因为上下文向量c是直接用所有词向量x1:x_m算出来的。
#### 总结Attention Layer
- Attention layer: $$C=\mathrm{Attn}(X,X^{\prime}).$$
	- 其中X为Encoder输入，X'为Decoder输入
	- 有三个参数矩阵: $$W_Q,W_K,W_V.$$
<img src="imgs/Pasted image 20250416163957.png">

#### Self-Attention without RNN
- Self-attention layer: $$C=\mathrm{Attn(X,X)}.$$
>Self-Attention的两个输入是相同的矩阵X，也就是说自己和自己比较得到c；而Attention是X‘与X比较。


<img src="imgs/Pasted image 20250416164422.png">
每个x通过三个参数矩阵计算Query,Key,Value，然后Q和K计算权重，并对V加权求和得到对应的c。
<img src="imgs/Pasted image 20250416164912.png">
Weights: (m个m维向量)

$$\alpha_{:1}=\mathrm{Softmax}(\mathbf{K}^Tq_{:1})\in\mathbb{R}^m$$
Contex Vector: 

$$c_{:1}=\alpha_{11}v_{:1}+\cdots+\alpha_{m1}v_{:m}=V\alpha_{:1}.$$

最后Self-Attention Layer输出m个c向量

$$c_{:j}=V\cdot\mathrm{Softmax}(K^Tq_{:j}).$$

# Summary
- Attention最初是用在Seq2Seq模型中解决遗忘问题(2015)
- 后来发现Attention可以用在所有RNN上，也就是Self-Attention (2016)
- 2017：那篇论文发现，根本不需要RNN，只要Attention就可以了。