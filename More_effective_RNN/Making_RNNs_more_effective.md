提升RNN表现的四个技巧
# 1. Stacked RNN
>全连接层可以堆叠，卷积层也可以堆叠，那么RNN也同样可以。

可以把**多层RNN堆**叠起来，构成多层RNN网络；例如使用三层RNN，第一层的输出h是第二层的输入，最后第三层的RNN的状态向量是最终的输出。
<img src="Pasted image 20250409211505.png">

## 代码
需要三个LSTM层，其中前两个需要设置`return_sequences=True`，这是因为它们的输出需要返回作为下一层的输入。

# Bidirectional RNN
RNN与人的阅读习惯相同，从前往后读积累信息。但其实RNN也可以从后往前读。
<img src="Pasted image 20250409212741.png">
- 两个不同方向的RNN
- 将它们的状态向量连接得到y；如果还有下一层，作为下一层的输入；
- 如果只有一层，将y丢掉，只保留两个RNN最后的$[h_t,h_t^\prime]$ 作为输入文字中抽取的特征向量，来判断电影评论的正面或者负面。
双向RNN总是比单向效果好，可能是反向的RNN能够更好地记忆前面的文字。
## 代码
```python
from keras.layers import Bidirectional

model.add(Bidirectional(LSTM(state_dim, terturn_sequences=False)))
```

# Pretrain
>预训练在深度学习中非常常用。如果网络太大而训练集不够大，可以先在大数据集上做预训练，让神经网络有比较好的初始化，也可以避免overfitting。

比如我们只有2w个数据，Embedding层却有32w参数，所以会overfitting。可以对Embedding层做预训练
1. 在大数据集上训练模型，最好是接近的任务，让Embedding层可以学习到词的情感
2. 把其他层丢掉，只保留embedding层；搭自己的模型，RNN和全连接层随机初始化，而使用之前训练好的Embedding层参数
3. 固定Embedding层的参数，只训练其他层

# 总结
- SimpleRNN和LSTM是两种RNN
- SimpleRNN很容易遗忘，效果不好；LSTM的记忆比SimpleRNN长很多，所以应该用LSTM。
- 有一些方法能让RNN效果更好，例如使用双向LSTM，不会比单向差。
- RNN层可以累加起来搭成深度神经网络，容量比单层更大，当数据较大时效果更好。
- 预训练：当数据少的时候，Embedding层容易overfitting，可以使用预训练。