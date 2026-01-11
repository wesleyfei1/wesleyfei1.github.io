### core understanding

- **problem definition**(one sentence): In order to draw  understanding from the whole sentence instead from "some part of it" and to compute more quickly
- **current solutions**(现有方法总结)：current sequence transduction models rely heavily on RNN & CNN,low in computing efficiency.
- **核心思想**(可以应用的思想)：

use "Multi-head" techniques to boost training

Attention mechanism: $Attention(Q,K,V)=softmax(\dfrac{QK^T}{\sqrt{d_k}})V$

- **solution**细节(focus on module):

![image-20251013093915444](D:\wesleyfei1.github.io\blogs\Sophomore-Fall\论文阅读\NLP_2017_AttentionIsAllYouNeed.assets\image-20251013093915444.png)

![image-20251013093944736](D:\wesleyfei1.github.io\blogs\Sophomore-Fall\论文阅读\NLP_2017_AttentionIsAllYouNeed.assets\image-20251013093944736.png)

**attention**:

Self attention: V=K=Q=x(input)

Cross attention: K=Q=x(encoder),V=x(decoder)

Masked attention:  K is a Lower triangular matrix

FFN: $FFN(x)=Fc(ReLU(Fc(x)))$

still use a sequence model, but use attention as its decoder & encoder



**Positional Encoding**:![image-20251013094336023](D:\wesleyfei1.github.io\blogs\Sophomore-Fall\论文阅读\NLP_2017_AttentionIsAllYouNeed.assets\image-20251013094336023.png)



- **experiments**:

数据集，baselines,对于关键的图和表，性能提升，稳定性

data: WMT 2014 English-German dataset, 8 P100 GPU,Adam optimizer(learning rate warmup)

dropout=0.1

Lable smoothing: L2 Regularization $\epsilon=0.1$

 ablation experiment: 

N larger, better（deeper, better)

training longer ,better

dropout better

regularization: better（but little)







- **will I use it**: Multi-head is very important, attention will use the ability of GPU greatly
- **how can I use it**: this question don't need it

### critical-thinking

- **problem**: 如果我们见到这个问题，我们的想法是什么
- **solution**: assumption是什么？是否可以有所改进

softmax? attention best? and the benchmark didn't improve greatly, positional coding???

- **why attention**:

attention can sum up all the information in a sentence within O(1)



but the architecture: N*(Attention block + FFN) as out encoder & decoder

- **flaws**：论文中的可能的问题

### creative-thinking

- some interesting thoughts

can we rethink attention from infromation theory's perspectiv