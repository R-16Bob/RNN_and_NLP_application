# Bidirectional Encoder Representations from Transformers (BERT)
BERT\[1\]的目的是预训练Transformer的Encoder网络，从而大幅提高准确率。

\[1\][BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

1. Predict masked word. 随机遮挡一个或多个单词，让Encoder预测被遮挡的单词
2. Predict next sentence. 把两句话放到一起，让Encoder判断是不是相邻的两句话

## Predict masked word
<img src="imgs/Pasted image 20250420163651.png">

第一个任务是预测被遮挡的单词
- "The __ sat on the mat"
<img src="imgs/Pasted image 20250420163902.png">
方法是将某个输入遮住变为MASK符号，进行embedding，得到 $$U_M$$ 
，因为Encoder是多对一，也就是U_M中包含整句话的信息，就能够预测被遮住的单词。

在U_M后面接一个softmax分类器，输出预测的概率分布p。
- e为被遮住词的one hot
-  $$Loss = \mathrm{CrossEntropy}(e,p)$$
- 使用梯度下降更新模型参数。
### 总结
预训练不需要标注数据，可以用百科等训练集，可以自动生成数据集。这样数据要多少有多少，可以训练出一个非常大的模型。

## Task2: Predict the next Sentence
- Given the sentence: "calculus is a branch of math."
- Is this the next sentence? 
	- "it was develpoped by newton and leibniz"
	- “panda is native to south central china”
判断是否是两句相邻的话？

- Input:
	- \[CLS\] "calculus is a branch of math."
	- \[SEP\] "it was develpoped by newton and leibniz"
- CLS是分类用的占位符
- SEP是分割句子的意思
- Target: Ture (or false)

50%的第二句话是训练数据中真实的。另外50%是从训练数据中随机抽取的。

<img src="imgs/Pasted image 20250420170234.png">

输入是增加CLS和SEP符号的两句话。CLS对应的输出c接一个二分类器，输出分类结果。
- 然后使用梯度下降更新Embedding和Transformer的参数
- 这样的作用是让参数包含上下句中的关联

## Combining the two methods
BERT将两个任务结合起来用来预训练Transformer

- 把两个句子拼接起来，然后随机遮挡15%的单词。
<img src="imgs/Pasted image 20250420170652.png">

有三个任务就有三个损失函数（一个二分类，两个交叉熵）
- 目标函数是三个损失函数的加和
- 通过梯度下降更新参数

## 总结
- BERT的好处是不需要人工标注数据。两种任务的标签都是自动生成的
- BERT使用了英文维基百科，长度是2.5billion个单词

### 计算代价大
- BERT Base
	- 110M 参数
	- 16 TPUs, 4 days for training
- BERT Large
	- 235M 参数
	- 64 TPU, 4days for training

预训练成本极高！但是可以直接下载预训练好的参数