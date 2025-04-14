# 机器翻译与Seq2Seq模型
使用Sequence-to-Sequence模型把英语翻译为德语。
- Many to Many问题，输入和输出长度都不固定

## 数据集
[Tab-delimited Bilingual Sentence Pairs](https://www.manythings.org/anki/)
- 一句英语对应几句德语，只要匹配其中一条就算翻译正确
- 预处理：大写变成小写，去掉标点符号

## 1. Tokenization(词语切分) & Build Dictionary
把一句话变成很多单词或字符
- 对两种语言使用两个不同的tokenizers
- 构建两个字典
Tokenization可以是char-level或者word-level，这里为了简单使用char-level

### 为什么需要两个不同的tokenizers和字母表？
- 在char-level，两种语言有不同的字母表
<img src="Pasted image 20250414130854.png">
- 对word-level，两种语言也有不同的vocabulary（词汇表）

结果是得到了26个字母+空格的len27英文字典和额外增加start sign(\t)和stop sign(\n)的len(29)德语字典。

## 2. One-Hot Encoding
- 通过Tokenizer将句子切分为char后，使用字典进行Encoding，可以把一句话变成index的列表（Sequence）；
- 最后，将Sequence用One-hot向量表示，一句话就可以用矩阵表示。

## 搭建并训练Seq2Seq 模型
### Seq2Seq模型
有一个Encoder编码器和一个Decoder解码器
- Encoder是一个LSTM，只输出最后的状态h和传送带c给Decoder
- Decoder其实就是一个文本生成器，唯一的区别是初始状态；上节的初始状态是全0向量，而这一节的初始状态是**Encoder的最后状态**，从而得知英语的内容；
- Decoder是一个LSTM，每次接受一个输入，然后输出对下一个句子的预测
	- 第一个输入必须是起始符
<img src="Pasted image 20250414133051.png">
- 有了标签后可以通过交叉熵损失函数进行反向传播，更新Encoder与Decoder
- 这样每次输入增加一个字符进行训练；最后一轮将整句德语作为输入，将停止符\n作为标签y
<img src="Pasted image 20250414133551.png">
模型的实现如上图：Encoder输入英语的One-hot矩阵输出最后的状态(h, c)作为Decoder的初始状态，把Encoder与Decoder连起来能够进行反向传播；Decoder的输入是德语句子的一部分，输出状态向量h'，并通过全连接层输出下一个字符的预测。

## 模型推理
1. 首先将英文句子输入Encoder，得到初始状态(h0, c0)
2. 将起始符作为Decoder的输入，得到状态(h1, c1)和下一个字符的预测
3. 将下一个字符作为Decoder的输入，得到新的状态和下一个字符预测；不断重复直到预测的下一个字符是终止符。此时停止生成，输出生成的所有字符串。
### 代码实现
在数据预处理遇到问题，暂时没能成功。我认为是因为下载的数据太大了，有40M！转换为矩阵过大，我的计算机内存不足。

## 如何改进Seq2Seq?
### 1. 用双向LSTM代替单向LSTM
因为Enccoder把英文的信息提取到输出的状态向量中。但如果英语句子很长，LSTM就能遗忘，状态向量就没法提供完整信息，所以可以用双向LSTM代替，增加记忆。因为反向的链可以记住正向链忘记的文本开头的信息。

### 2. 使用word-Level Tokenization
使用char-level比较方便，不需要Embedding层，但最好还是word-level。
- 英文单词长度平均为4.5c，如果用word而不是char，输入序列可以短4.5倍！序列更短就更不容易遗忘
- 但是，想用word-level需要有大的数据集：词汇表大约10,000，需要word Embbeding转换为低维词向量
- embbeding的参数数量太大了，小数据集无法训练embbeding层，容易overfitting

### 3. Multi-Task Learning
可以进行多种翻译任务，这样Encoder的训练数据变多，效果会更好。

### 4. Attention
见下节