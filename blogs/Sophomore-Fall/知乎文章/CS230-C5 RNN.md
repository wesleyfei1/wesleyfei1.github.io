CS230的最后一块内容是介绍关于RNN的，个人感觉Andrew Ng在RNN网络的介绍上非常详细，但是对于NLP&LLM部分就显得不太行了，本文中我们将会就RNN网络的来龙去脉进行介绍，对于RNN的应用我们将会留到cs224n中进行介绍

## 



## RNN应用

RNN第一个应用就是对于一段语言的处理，而对于语言的处理，即为每一个单词的表示形式从最一开始的**独热编码**变为**Word Embedding**。

对于变成了word Embedding之后的处理我们现在使用的是Transformer。

- self attention:其中(Q,K,V)可以认为是我们对于每一个单词的理解认为是一个查询，而查询会影响到我们对于这个单词的理解方式，最后加上相应的权重V,即为$$Attention(Q,K,V)=softmax(\dfrac{QK^T}{\sqrt{d_k}})V$$这里的（Q,K,V)为对于输入单词经过相应的矩阵$$W^Q,W^K,W^V$$产生的。

  self attention的输入是**单词的embedding**，而输出为在经过attention处理之后的应该被认识的方式(**每一个单词对应着一个向量**)

- Multi-Head attention，类似于多重卷积核，对于self attention我们使用多个，就可以得到对于每一个单词的不同的理解方式，，最后经过一个全连接，得到**对于每一个单词的理解方式(一个向量)**

- Transformer:我们不考虑add&norm，即为

**Encoder部分**：单词输入，经过multi-hear attention之后得到一些输出

**Decoder部分**：输入为之前的单词，Encoder对于单词的理解，经过Attention后的一些基本知识，以上的部分经过Attention之后concat起来，最后经过一个softmax得到输出。

具体的部分我们留到cs224n中详细讨论

---

而对于word Embedding而言，我们知道了它是非常重要的，每一个单词对应的向量就表示了它的意思在不同维度的比重。

而训练word Embedding的数据集的构建为从一段文字中挑选的若干个，对于这一块的训练过程我留到CS224n去讨论。

---

机器翻译的评判标准我们会使用Bleu score，简单而言就是分别考察1个单词，2个单词。。。多个单词预测成功的概率并进行奖励惩罚。

