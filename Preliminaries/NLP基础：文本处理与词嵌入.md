原视频链接：[RNN模型与NLP应用](https://www.bilibili.com/video/BV1w54y1L7xK?spm_id_from=333.788.videopod.sections&vd_source=e63f08e3795a7d51a7cfc6c0294d87ee) by [ShusenWang](https://space.bilibili.com/1369507485)

# 一、数值处理基础
## 处理Categorical 特征
- 对于数值类型，例如年龄可以直接用
- 对于二元的类别特征，如性别，可以用1和0表示
- 若是国籍这种类别特征，使用one-hot
	- 197个国家，对应197维的one-hot，例如中国用[1,0,0,...,0]表示；**使用one-hot而不是数值是因为，如果数值相加时，中国+美国=印度这是不合理的；** 而如果是one-hot相加，中国+美国=同时有两个国籍，这就合理了
## 处理文本特征
1. Tokenization（将文本分割为单词）
2. 统计词频，然后将单词按照词频递减排列，将词频换为index，词频最高的为1；（统计词频的目的是保留常用词，删掉低频词），这是因为：
	1. 低频词通常没有意义：例如名字还有拼写错误
	2. vocabulary越大，one-hot的维度越大，计算越慢，在word-embedding layer有越多参数
3. One-Hot Encoding

# 二、文本处理与词嵌入

## IMDB电影评论数据
- 50k电影评论
- 标签为positive或者negative，也就是**二分类问题**
- 25k for training and 25k for test
- Download from （实际上可以通过代码下载）
	- http://ai.stanford.edu?~amaas/data/sentiment
	- http://s3.amazonaws.com/text-datasets/acllmdb.zip
## Text to Sequence 文本处理：分词、index、序列
### Step 1：Tokenization
将text分割为tokens（单词或字符）列表
```python
texts[i] = "the cat sat on the mat."
tokens[i] = ["the", "cat", "sat", "on", "the", "mat"]
```
一些问题：
1. 是否将大写转换为小写？Apple与apple含义不同
2. Remove stop words, eg., the, a, of 这些词对二分类没有帮助 
3. Typo correction
### Step 2: Build Dictionary
使用字典统计词频，将word映射为index。
### Step 3: One-hot Encoding
这样一句话可以用正整数的列表表示，称为Sequences（序列），如果有必要，可以进一步映射为One-hot
```python
token_index = {"the":1, "cat":2, "sat":3, "on":4, "mat":5, ...}
sequence[i] = [1,2,3,4,1,5]
```
### Step 4：Align Sequences
由于评论有长有短，Sequence 列表的长度是不一样的，训练数据没有对齐(aligned)。
然而，机器学习需要把数据存储在矩阵或张量里，要求每条序列都有相同的长度。
**解决方法**：
- 只保留w个词（最前或最后）
- 如果序列比w短，做zero padding
现在，每个词都用一个正整数来表示
## Word Embedding： Word to Vector 词嵌入
如何将单词用向量表示？ One-hot
- 假设字典里有v个单词，就使用维度为v的one-hot向量表示
如果有1w个词，输入向量就是1w维的，我们不希望这样，所以
## Second, map the one-hot vectors to low-dimensional vectors by 参数矩阵P
将one-hot向量映射为低维向量。方法是乘一个dxv的参数矩阵P
<img src="Pasted image 20250327204230.png">
关于参数矩阵P：
- d行（词向量的维度，用户决定），v列；可以把每列看作对应e的一个词向量
- 参数矩阵从训练数据中学习出来（也就是所谓的embedding层），学出的词向量会带有感情色彩：在向量空间表现为，积极的词语互相接近，而消极的词语远离积极的词语并且也相互接近

>代码部分！
>视频中使用了keras库作为深度学习框架，所以我新建了一个conda环境。除了keras库之外还需要TensorFlow
>- 然后tensorFlow安装失败，才发现人家要求python3.6-3.9，而conda默认最新的是3.11
>- 总之，新建环境前，要提前了解使用库需要的python版本

字典大小v-10k, 词向量维度d=8, 输入长度=20
embedding得到（20， 8）的矩阵；embedding层的参数数量是80000，即dxv，也就是参数矩阵P

## Logistic Regression for Binary Classification
详见代码
