# Attention
## Seq2Seq的缺陷
如果输入的句子很长，状态向量就记不住全部输入的信息了。
如果Encoder把一些信息忘记了，Decoder就无法知道那些信息，从而无法生成正确的翻译。
<img src="Pasted image 20250414152957.png">
>BLEU主要是用来衡量机器翻译系统输出的翻译文本与参考翻译（通常是人工翻译的高质量文本）之间的相似度。BLEU分数的取值范围是0到1。0表示机器翻译结果和参考翻译之间没有任何匹配，而1表示机器翻译结果和参考翻译完全一致。BLEU主要基于n - 元组（n - grams）匹配来计算分数。n - 元组是指文本中连续的n个词组成的序列。它会计算机器翻译结果中与参考翻译匹配的n - 元组的数量。通常会考虑不同长度的n - 元组，如一元组（单个词）、二元组、三元组等。

## Attention for Seq2Seq Model
- Attention极大地提升Seq2Seq模型
- Decoder每次更新状态时，会再看一遍Encoder所有状态，这样就不会遗忘
- Attention还能让Decoder知道应该关注哪里
- Attention能大幅提升准确率，缺点是计算量非常大。
### Attention原理
当Encoder结束工作后，Attention与Decoder同时开始工作。
- Encoder的每一个状态都需要保留下来，并计算与最后状态s0的相关性
<img src="Pasted image 20250414153812.png">
Weight: $$\alpha_i=\mathrm{align}(\mathbf{h}_i,\mathbf{s}_0)$$
将相关性alpha称作权重，每个alpha都是0到1的实数，所有alpha和为1.

#### 如何计算s0与hi的相关性
方法一：第一篇论文提出
<img src="Pasted image 20250414154420.png">
将hi与s0连接，乘矩阵W得到向量；经过双曲正切映射到(-1,1)，再与向量V求内积，最终得到一个数。用softmax正则化为0-1之间和为1的alpha。

方法二：更常用，与Transformer相同
1. Linear maps
$$k_i=\mathbf{W}_k\cdot h_i,$$
 for i=1 to m.
 $$q_0=\mathbf{W}_Q\cdot s_0$$
 第一步是分别用两个参数矩阵W_k和W_q做线性变换，得到k_i和q_0两个向量
 2. Inner product
 $$\tilde{\alpha}_i=k_i^Tq_0,$$
 for i=1 to m.
 第二步是ki和q0做内积，得到m个$\tilde{\alpha}$
 3. Normalization
$$[\alpha_1,\cdots,\alpha_m]=\mathrm{Softmax}([\tilde{\alpha}_1,\cdots,\tilde{\alpha}_m])$$
第三步对m个$\tilde{\alpha}$做softmax变换，得到m个alpha

### Context vector
将得到的权重与对应h相乘并相加，得到上下文向量c_0
$$c_0=\alpha_1h_1+\cdots+\alpha_mh_m.$$
### Decoder状态s的更新
SimpleRNN:
$$s_1=\tanh \left( \mathbf{A^{\prime}}\cdot\begin{bmatrix}
 \mathbf{x}_1^{\prime} \\
 s_0
\end{bmatrix} +b\right )$$
SimpleRNN+Attention:
$$s_1=\tanh \left( \mathbf{A^{\prime}}\cdot\begin{bmatrix}
 \mathbf{x}_1^{\prime} \\
 s_0\\
 c_0
\end{bmatrix} +b\right )$$
<img src="Pasted image 20250414161450.png">
一共计算了多少次权重?
Decoder的每一个状态都需要重新计算m个权重，所以时间复杂度是mxt，很高。
**Attention避免遗忘，大大提升了准确率，代价是巨大的计算**

<img src="Pasted image 20250414162027.png">
关于Attention做的事情，其实是根据Decoder当前的状态，帮助Decoder找到Encoder中应该关注的状态，从而让Decoder生成正确的状态，做出正确的预测

## 总结
- Seq2Seq模型Decoder基于当前状态产生下一个状态，可能已经遗忘Encoder的部分输入
- 如果使用Attention，Decoder在产生下一个状态时会看一遍Encoder的所有状态，避免遗忘
- 除了解决遗忘问题，attention还能告诉Decoder应该关注Encoder的哪一个状态
- Attention的缺点是计算量很大，标准Seq2Seq的时间复杂度是O(m+t)，而Seq2Seq+Attention是O(mxt)