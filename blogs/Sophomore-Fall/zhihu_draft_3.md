# MIT 6.S184 生成式人工智能——Diffusion & Flow Model 简介

对于diffusion & flow model的从数据z  pdata中去学习的过程，下面来做一个小小的整理

1. 首先对于我们的基本概念就是
   - X~p init, Z~p data分别表示我门的噪声数据以及我们的需要从中去学习的数据
   - conditional & marginal:对于z~p这个东西，我们将所有的东西打散，然后用全概率公式的类推去执行
   - pt(x)=Xt表示在这个时刻的物体的”坐标“
2. 我们的一切的宗旨是对于每一个z学习到对应的特征，然后通过积分或者别的方式合成到整体上。
3. 第一步，假设pt(x|z)=???(conditional probability path),即为p0(x|z)=p init, 而p1(x|z)=delta z(完全学会了总体的特征)
4. 通过pt(x|z)=F(xt,z),得到Xt和Xinit以及Z的关系
5. 由此带入到SDE&ODE的方程中
6. 得到我们希望训练达到的模型在我们**随机抽取的x,t,z**的条件下，达到的效果。
7. 使用梯度下降

对于score function:噪声带来的扩散效应下的反向校正

需要求解的量

pt(x|z),utarget(x|z),score funtion

## 在编程求解的时候我们需要注意的问题

在类的继承的时候各种奇奇怪怪的操作

对于实际上我们向量化的操作，每一次随机选取batch_size大小的东西随机去抽取，然后进行

```python
for loop:
    select x~pinit,z~pdata,t~[0,1]randomly
    Loss=|utheta-utarget|2
    gradient descent
    keep it
when showing the results& making predictions on the results:
    p=(conducting SDE & ODE based on the utheta(x) that we just trained)
    draw the line on the picture
```

通过以上的方法，我们就可以让utheta学习到zdata中的数据。

对于生成数据



# About werewolves_arena

[article website](https://arxiv.org/pdf/2407.13943)

## Try to accomplish

how to use werewolves_arena to evaluate the performances of different LLMs. Especially on dynamic turn-taking systems and the evaluating principle

## Key elements

bidding system(mask 0,1,2,3,4 to show how ergent the LLM model want to speak)

evaluation system(arena evaluation& seer evaluation)

**focused on when & how to speak(simulate human's actions better)**

## what can I use

bidding system

more stuff that can be used to evaluate the performances of a LLM system(by game)

## References

more on evaluation on LLM

[Github repo](https://github.com/google/werewolf_arena)
