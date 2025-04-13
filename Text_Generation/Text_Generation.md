# RNN应用：文本生成
如果用莎士比亚的书训练RNN，RNN就会生成莎士比亚风格的语言。

## 原理
Input: "the cat sat on the ma"
将字符转换为one-hot，输入RNN，最后一个状态h乘一个矩阵W得到一个向量，然后使用softmax转换为每个字符的概率分布（W是全连接层）。

<img src="Pasted image 20250413212045.png">
<img src="Pasted image 20250413212304.png">

可以选择概率最高的字符（或者随机抽样），也就是t. 将t放到句子末尾，并将新的句子输入RNN，从而获得下一个字符的预测。

## 如何训练RNN
将文本的片段(segment)（有重叠）作为**输入**，将紧接着的下一个字符作为**标签**。并设置步长让片段向后移动。
- 例如：seg_len=40表示输入的分段长度为40；stride=3表示下一个分段向后移三个字符
- 训练的目的是输入一个片段，预测下一个字符。其实就是一个多分类问题

## 实战文本生成
见Alice_Generator
我用《爱丽丝梦游仙境》训练了一个LSTM，生成爱丽丝风格的文本。

种子文本：
Alice was captured by Queen when the rabbit says: 'why do cr
Alice was beginning to get very tired of sitting by her sist

### 下一个字符的抽样方式
首先，输入60长度的输入，得到下一个字符的softmax预测。
问题：如何确定下一个字符？
1. greedy selection
	1. 那个字符概率最大就用哪个字符：`next_index = np.argmax(pred)`
	2. 生成是完全确定的；不好，希望得到更加多元的结果。
2. 从multinomial distribution采样：完全根据概率生成
	1. `next_onehot=np.random.multinomial(1, pred, 1)`
	2. `next_index = np.argmax(next_onehot)`
	3. 由于按照预测概率选择，太过随机，容易发生拼写错误
3. 用multinomial distribution调整概率值;**用到[0,1]之间的temperature**
	1. `pred = pred ** (1/temperature)`
	2. `pred = pred / np.sum(pred)`
**关于温度temperature**：假设 `pred` 是一个概率分布，表示模型对下一个词的预测概率。`temperature` 是一个(0.1]的正数，通过调整 `temperature` 的值，可以改变概率分布的“尖锐度”。
- 当temperature为1时，概率不变
- 当temperature<1时，1/temperature大于1，且温度越指数越大，从而让概率值更尖锐。也就是概率大的被放大，概率小的被缩小。当温度很低，相当于确定性选择；温度高时，则更偏向随机选择。

## 总结：
### 如何训练神经网络
1. 将文本划分为(segment, next_char)而元组
2. One-hot encode 
	1. Chareacter->vx1 vector
	2. segment -> lxv matrix
3. Build and train a NN
	1. lxv matrix==>LSTM==>Dense==>vx1 vector （softmax的概率）
### 如何实现文本生成
1. 提供种子segment
2. 重复下面的步骤
	1. 将one-hot编码的segment输入神经网络
	2. 神经网络输出概率向量
	3. 根据概率做抽样得到下一个字符
	4. 将下一个字符添加到segment，得到下一个segment