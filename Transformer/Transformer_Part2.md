# Multi-Head Attention
## 回顾Single-Head Self-Attention
<img src="imgs/Pasted image 20250416164422.png">
这样的Attention Layer被称为单头(Single Head)注意力层
## 多头Self-Attention 
是由l个single-head self-attentions组成的，它们不共享参数。
- 每个单头自注意力有3个参数矩阵W_Q, W_K, W_V
- 所以多头注意力有3l个参数矩阵
<img src="imgs2/Pasted image 20250420150755.png">
把单头的输出堆叠起来（Concatenation），作为多头的输出
- 假如单头的输出是dxm的矩阵（Context Vector的维度为d，一共m列）
- 那么多头的输出就是(ld)xm的矩阵

## 多头Attention
可以用l个Self-Attention构建多头Self-Attention，同样可以用l个Attention构建多头Attention。
<img src="imgs2/Pasted image 20250420151335.png">
同样把单头的输出堆叠起来，就是多头的输出。

# Stacked Self-Attention Layers
接下来用多头Attention和Self-Attention搭建深度神经网络。
## Self-Attention Layer+Dense
<img src="imgs2/Pasted image 20250420151747.png">

在多头Self-Attention的输出后加一个全连接层Dense，得到m个输出向量u；
注意**所有全连接层都相同，矩阵W_U**。

可以在输出u后面再接一个Multi-Head Self-Attention Layer，想搭多少层都可以，原理与多层RNN是一样的。

## Transformer's Encoder

Encoder网络的一个Block是这样：
<img src="imgs2/Pasted image 20250420153512.png">
输入词向量维度是512，一共m个，输入的是512xm的矩阵，经过多头注意力和Dense输出的也是一样。
>我想Dense的矩阵是512x(512xl)，这样输出矩阵的大小不变。

然后是Encoder的设置。
### Skip-Connection
使用了ResNet的Skip-Connection
- 在ResNet（残差网络）中，Skip - Connection是一种特殊的连接方式。它允许输入信号绕过一些层（通常是两个或多个卷积层）直接传递到后面的层。具体来说，对于一个残差块（Residual Block），输入数据 x 经过一系列的卷积操作（如先经过一个卷积层，再经过一个批量归一化层，然后是激活函数，接着又是卷积层等一系列操作）得到的结果为 F(x)。同时，输入数据 x 会通过Skip - Connection直接加到 F(x) 上，最终的输出为 H(x)=F(x)+x。
- 这种结构的核心思想是基于残差学习。假设理想情况下，网络能够学习到输入和输出之间的映射关系 H(x)，那么可以将其表示为 H(x)=F(x)+x。其中 F(x) 就是残差，即输入和输出之间的差异部分。这样设计使得网络可以更容易地学习到残差，而不是直接学习整个映射关系。
- **Transformer架构本身并没有直接使用ResNet的Skip - Connection，但它有类似的机制**
    - 在Transformer的编码器（Encoder）和解码器（Decoder）模块中，使用了“残差连接”和“层归一化”的组合。以编码器中的一个层为例，输入数据首先会经过一个多头自注意力（Multi - Head Self - Attention）模块，得到一个输出。然后这个输出会和输入数据进行残差连接，即 Output=Input+Attention(Input)。之后会进行层归一化操作。
    - 这种设计的原理和ResNet的Skip - Connection有相似之处。它也是为了保证信息的传递和梯度的稳定传播。在Transformer中，多头自注意力模块可能会对输入数据进行复杂的变换，通过残差连接可以保留输入数据的一部分原始信息，防止信息在变换过程中丢失。同时，这种机制也有助于在训练过程中保持梯度的稳定，使得Transformer能够有效地处理长序列数据，训练更深的网络结构。

<img src="imgs2/Pasted image 20250420154512.png">
总之，Skip-Connection的意思是把输入加到输出上，让神经网络学习残差（也就是输入与输出的差值），从而的深层网络更容易学习同时解决梯度问题。

Transformer的Encoder总共有6个Block，每个Block有两层（多头注意力和Dense）
- Block之间不共享参数
- 最终输出仍然是512xm的矩阵，即m个输出向量u，大小与输入矩阵相同。

# Stacked Attention Layers
## Stacked Attentions
现在搭建Decoder的Block.
Block的第一层是多头Self-Attention，输出是512维的c1到c_m。

<img src="imgs2/Pasted image 20250420155530.png">

第二层是多头Attention，输入是两个序列，u1到 $u_m$ 和c1到 $c_m$ 
最上面再加一个Sense，得到:

$$\mathbf{s}_{:1}=\mathrm{ReLU}(\mathbf{W}_Sz_{:1})$$

所有Dense共享参数
<img src="imgs2/Pasted image 20250420155912.png">

总结：Decoder的Block有三层，分别是多头自注意力层、多头注意力层和全连接层。
- 它有两个输入序列和一个输出序列，输出序列的长度t与 $x^{\prime}$ 的长度相同，因为Attention是这样计算的。
<img src="imgs2/Pasted image 20250420160357.png">

## Put Everything Together
Encoder网络很简单，叠加6个Block，每个Block有两层，分别是自注意力和全连接，输入输出矩阵大小相同。

Decoder的一个Block则是如上图所示。可以在之上继续堆叠Block，输入为仍然为Encoder的输出，第二个序列输入为上一个Block的输出。

<img src="imgs2/Pasted image 20250420160916.png">
Decoder一共有6个Block，每个Block有三层，分别是自注意力层、注意力层和全连接。

### 回顾RNN Seq2Seq模型
<img src="imgs2/Pasted image 20250420161205.png">
输入是两个序列，输出是一个序列。
- 可以知道RNN Seq2Seq模型与Transformer的输入输出完全一样，因此，RNN Seq2Seq能做的，Transformer都能做。

# Summary
- 将单头注意力组合为多头注意力。
<img src="imgs2/Pasted image 20250420161453.png">
- Encoder网络就是由多头自注意力层与全连接层搭出来的。
- Decoder网络用了多头自注意力层、多头注意力层和全连接层。
