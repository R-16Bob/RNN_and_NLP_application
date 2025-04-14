# 自注意力\[2\]
第一篇Attention的论文\[1\]用在Seq2Seq中，用于解决遗忘问题。其实Attention不止能用在Seq2Seq，Attention能够用到所有RNN上。
\[1\][Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
\[2\][ Long Short-Term Memory-Networks for Machine Reading](https://arxiv.org/abs/1601.06733)

## SimpleRNN + Self-Attention
SimpleRNN:

$$h_1=\tanh \left( \mathbf{A}\cdot\begin{bmatrix}
 \mathbf{x}_1 \\
 h_0
\end{bmatrix} +b\right )$$

SimpleRNN+Self-Attention:

$$h_1=\tanh \left( \mathbf{A}\cdot\begin{bmatrix}
 \mathbf{x}_1 \\
 c_0
\end{bmatrix} +b\right )$$

用c_0取代h_0，其他都一样

### 计算新的上下文向量
context vector是之前状态向量的加权平均。
- 忽略全0的h0，就只剩下一个状态向量h1，因此c1=h1
之后，想要计算c_2就需要计算权重:

$$\alpha_i=\mathrm{align}(\mathbf{h}_i,\mathbf{h}_2)$$

即当前状态h2与之前状态(h1,h2，因为h0为全0向量，忽略)的相关性，从而得到新的权重，加权平均得到c2。

## 总结
- Self-Attention能够解决RNN遗忘的问题。在每一次更新状态前，看一眼之前的所有状态，这样就不会遗忘之前的信息。
- Self-Attention不局限于Seq2Seq模型，而是可以用到所有的RNN中。具体来说，前者是Decoder查看Encoder的状态，后者则是RNN查看自己过去所有的状态，所以叫自注意力。
- RNN不仅能避免遗忘，还能够帮助RNN找到当前状态最相关的词是哪些，如下面论文插图所示。红色是输入，蓝色表示权重的大小。
<img src="Pasted image 20250414171252.png">
