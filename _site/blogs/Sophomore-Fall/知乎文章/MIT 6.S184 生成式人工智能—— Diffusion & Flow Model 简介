# MIT 6.S184 生成式人工智能—— Diffusion & Flow Model 简介

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

## 对于$$u_t^{\theta}(x)$$的网络结构的设计以及前端，后端的设计，conditonal generation

首先是对于中间对于我们的更新函数而言，一方面我们可以使用多个encoder,decoder并且在中间加上一些小小的mid-coder的形式（从U-net中得到的想法）



对于我们的前端，如何将一段话变成y变成一串的数字，我们会使用clip的方式

stable diffusion: clip&t5-xxl等embedding的方式，同时使用MM-DiT的方式

Meta Gen Video:3 embeddings, cross&self attention



对于conditonal generation部分而言：ut&~ut, 对于取样时从（z,y)中获得的，然后通过网络计算得到u_theta，最后更新的时候使用别的方法。

具体的证明是使用了classifier-free guidance.的fang'fa













生成式AI是目前的一个热点方向，类似于cs229中的无监督学习，都是通过训练数据集去学习到**如何进行预测/生成结果**,只是对于无监督学习中目标(将数据进行合理的分类/提取)，评估方法都有较为简单的数学形式，而本文将会介绍对于生成式AI的生成原理，基本模型，训练目标，训练预测过程。

## 生成式AI的生成原理

Gnerative AI，输入一串指令y，我们希望基于指令生成一张图片，那么就会得到一个X, 而如果我们让他再生成一张，常常会生成一张不一样的。

**目标**：由于我们的每一次“希望AI生成一张图片”相当于一次查询取样，之际上就可以认为我们是从一个概率分布中去抽取一个样本，这提示我们Generative AI的目标是生成一个**概率**。也就是对于所有的问题都要从概率论的角度去思考

**初始化**：实际上，如果我们进行任何的训练，就是一个啥都不会的模型，那么对于你输入任何的指令，它应该都只会输出一堆的噪声。即原始模型只会进行随机的输出

**数据集**：对于一张图片我们可以看成是由RGB三种颜色（3个通道的）大小为$$Height\times Weight$$的矩阵，即为一张图片$$\in R^{C\times H\times W}$$,对于视频就是加上时间的维度即为$$R^{T\times C\times H \times W}$$。因此我们的输出为**$$z\in R^d$$的一个向量**。

于是我们拥有的是$$(z_{sample},y_{sample})$$的训练数据，用**概率论**的思想，也是他们都是从我们希望的分布$$p_{data}(z,y)$$中抽取的样本。

因此对于生成式AI也就是：

- 通过巨大的训练数据集作为**随机抽样的样本**，作为希望得到的概率分布$$p_{data}(z,y)$$
- **训练目标**：将模型由初始的$$p_{init}$$，通过训练得到一个**变换关系**，使得对于任何的y，都可以得到$$\hat{p}_{target}(x,y)$$,而且希望**通过某种评估手段**，使得$$\hat{p}_{target}(x,y)$$与$$p_{data}(z,y)$$在该意义下尽可能的接近。
- 最后基于给的输入y，从$$\hat{p}_{data}(.|y)$$中取样得到生成的数据，进而转换为图像形式

我们对于$$p_{init}$$而言，可以**端到端**的学习$$p_{data}$$，但是在这门课程中，我们将会通过演化过程的角度，学习对于$$p_{init}$$是如何经过演化关系变成$$p_{data}$$,通过训练这个演化函数完成从$$p_{init}\Rightarrow p_{data}$$的转变

## 演化的数学表达：微分方程

所谓的概率分布的“演化” ,就是对于其**空间**的一个变化，对于任意$$X_0\in R^d$$根据某种演化关系$$X_t=f(X_0)$$，直到$$X_t\in R^d$$,如果$$X_0\sim p_{init}$$,那么有$$X_t\sim p_{t}$$

因此我们最好的描述工具为**微分方程**, 从简单考虑，使用一阶微分方程去进行描述，结合需要加上一定的随机性，我们使用**ODE(ordinary differential equations)**以及**SDE(stochastic differential equations)**去描述

### ODE以及Flow Models

**ODE**

以$$R^d$$空间上的任意一个点t=0时刻位置在$$X_0$$的东西，对于t时刻他的位置为$$\phi_t(X_0)$$，则有初始条件$$\phi_0(X_0)=x_0$$,对于时间我们考虑$$t\in [0,1]$$的范围

假设在t时刻物体的“速度”为$$u_t(\phi(X_0))$$,即为$$u: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d, \quad (x,t) \mapsto u_t(x),$$,则我们物体的“位移” X应该满足

$$\dfrac{\mathrm{d}}{\mathrm{d}t} \phi_t(X_0) = u_t(\phi_t(X_0))$$ **flow ODE**

$$\phi_0(X_0)=X_0$$ **flow initial conditions**

**Flow Models**

对于$$\phi$$这个函数就表示了空间的流动情况，而如果我们对于考察的对象是从一个分布$$X_0\sim p_{init}$$，那么对于$$\phi_t(X_0)$$也应符合某个分布。特别的，我们希望在t=1(终点)的时候$$\phi_1(X_0)\sim p_{data}$$,则有我们的Flow Models:（简单起见，记$$\phi_t(X_0)$$为$$X_t$$）

$$\dfrac{\mathrm{d}}{\mathrm{d}t} X_t = u_t^{\theta}(X_t)$$

$$X_0\sim p_{init}$$ 我们通过训练神经网络$$u_t^{\theta}$$希望在t=1时刻有$$X_1\sim p_{data}$$，这个模型就叫flow models

**模拟Flow Models过程**

由于flow models是一个分布到分布的变化，我们模拟的方法是**多次从$$p_{init}$$中取样并使用ODE进行模拟**

而我们的ODE中的时间是连续的，为了模拟这个过程，我们使用**Euler Method**来模拟:将[0,1]划分为h大小的时间段，对于这段时间内用$$\dfrac{X_{t+h}-X_t}{h}$$作为$$\dfrac{d}{dt}X_t$$

伪代码即为：

```python
## sampling from a flow model with Euler method
step1: set t=0, h=1/n
step2: draw a sample X0~pinit
for i =1,...n-1:
    X[t+h]=X[t]+h*ut(Xt)
    t:=t+h
return X1
```

总结即为使用**取样作为分布观察分布变化情况**



## SDE & Diffusion Models

在ODE中我们是使用一个$$\dfrac{\mathrm{d}}{\mathrm{d}t} X_t = u_t^{\theta}(X_t)$$作为更新函数，但是如果对于$$u_t^\theta，X_0$$确定了，那么所有的位置都确定了。

但是即便如此，我们任然希望能加上一点的不确定性，简单起见，我们考虑**布朗运动**

**布朗运动(连续形式的markov过程)**

对于一个布朗过程$$W=(W_t)_{0\leq t\leq 1}$$ 是一个随机过程，满足$$W_0=0$$且满足

- **独立增量**：对于任意的$$0\leq t_0\leq t_1 ...\leq t_n=1$$,增量$$W_{t_1}-W_{t_0},...W_{t_n}-W_{t_{n-1}}$$为相互独立的随机变量
- **正态增量**：对于任何的$$0\leq s \leq t \leq 1$$,有$$W_t - W_s \sim \mathcal{N}(0, (t-s)I_d)$$

即为任意小的时间段内，位置的变化都可以认为是一个正态分布。

想要模拟布朗运动，我们仍然可以将[0,1]划分成n个长h的区间，只是由于正态增量原则，我们对于一个区间内的增量为$$\sqrt{h}*\mathcal{N}(0, I_d)$$,即为

$$W_{t+h} = W_t + \sqrt{h}\, \epsilon_t,\quad \epsilon_t \sim \mathcal{N}(0, I_d) \quad (t = 0, h, 2h, \ldots, 1 - h)
\tag{5}$$

**SDE&Diffusion Models**

所谓的SDE,就是在ODE的每一步上加上由布朗运动产生的贡献

$$X_{t+h} &= X_t + \underbrace{h u_t(X_t)}_{\text{deterministic}} + \sigma_t \underbrace{(W_{t+h} - W_t)}_{\text{stochastic}} + \underbrace{h R_t(h)}_{\text{error term}}\\&=X_t+hu_t(X_t)+\sigma_t \sqrt{h}\epsilon_t,\epsilon_t\sim \mathcal{N}(0, I_d)$$

写成方程的形式即为$$\mathrm{d}X_t = u_t(X_t)\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t$$，如果我们的$$X_0\sim p_{init}$$那么就称之为**Diffusion model**

```python
## sampling from a diffusion model using Euler-Maruyama method
step1: set t=0, h=1/n
step2: draw a sample X0~pinit
for i =1,...n-1:
    draw a sample \epsilon ~ N(0,Id)
    X[t+h]=X[t]+h*ut(Xt)+sigma[t]\sqrt{h}\epsilon
    t:=t+h
return X1
```

使用SDE/ODE作为数学基础，通过训练$$u_t^{\theta}$$来拟合分布的演化$$p_{init}\Rightarrow p_{data}$$是我们的目标

---



## 建立训练的目标

通过第二部分，我们已经知道Flow Model$$\dfrac{\mathrm{d}}{\mathrm{d}t} X_t = u_t^{\theta}(X_t)$$以及Diffusion Model $$\mathrm{d}X_t = u_t(X_t)\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t$$,而接下来就是如何进行训练的问题

由于我们有的东西是数据集，即有大量的$$x$$样本，自然的，而$$u_t^\theta(x)$$为会给出一个输出，自然我们运用Supervised Learning的想法：

**希望造出一些标签数据**,从而将损失函数定义为$$\mathcal{L}(\theta,x,t) = \Sigma^m_{i=1}\| u_t^\theta(x_i) - u_t^{\text{target}}(x_i) \|^2$$ 就可以使用梯度下降了。

于是我们的目标就变成了**寻找一个对于training target的等式$$u_t^{target}$$**。

### 演化路径

我们建立该等式的中心思想是：为了寻找$$u_t^\theta(x)$$,我们使用全概率公式的想法，学习来自不同的$$z\sim p_{data}$$的结果$$u_t(x|z)$$，最后合并到$$u_t(x)$$上，从而我们需要考虑将考察对象$$p_t(X)$$,以及"演化路径"也进行类似的转换

- **条件演化路径**：对于任何的$$z\in R^d$$,我们的有$$p_0(\cdot \mid z) = p_{\text{init}}, \quad p_1(\cdot \mid z) = \delta_z$$ 那么称从$$p_0(|z)\Rightarrow p_1(|z)$$的路径为条件演化路径(conditional probability path)
- **边缘演化路径**：也就是对于$$p_{data}(z)$$上进行积分

$$z \sim p_{\text{data}}, \quad x \sim p_t(\cdot \mid z)  \Rightarrow  x \sim p_t ,\quad 
 p_t(x) = \int p_t(x \mid z) p_{\text{data}}(z)\, dz$$

对于这两者我的形象化的理解是：条件演化路径中$$p_t(.|z)$$可以认为是$$t=0,h,.....1$$可以认为是一个总线结构,其内部构成为

- 将$$p_0(x)$$随着z打散到$$p_0(x|z)$$（类似于广播机制）
- 然后通过某种操作对于$$p_t(x|z)$$不断进行更新，使得我们的结果更加靠近$$p_1(x|z)$$
- 最后得到对于每一个$$p_1(.|z)$$的一个权重
- 如果我们需要计算$$p_t(X)$$,那么将$$p_{data}(z)$$看成是一个mask层，对于此时的$$p_t(x|z)$$通过mask层最后得到$$p_t(x)$$

网络图形可以看成是



可以发现，对于边缘演化路径而言，当$$z\sim p_{data}$$的时候，有$$p_t(x)=\int_z\delta_z p_{data}(z)dz=p_{data}(z)$$

由于我们的$$p_{init}$$一般为**噪声数据**高斯分布$$\mathcal{N} (0,I_d)$$,对于我们想要变换到的z的空间上的每一个点，以一个类似于**”光锥“**的东西将$$p_{init}$$映射到这个点上，最后再做一个掩码层，就可以将取自于$$p_{data}$$的各种离散的数据z转换成可以用概率表示$$p_t(x)$$中







### 条件速度以及边缘速度

原文中是使用向量场去描述，但是我觉得使用**速度场**更加的形象。

由于我们一般而言是定义了从$$p_t(x|z)$$的形式，可以理解为**总线上的一根支线，最后使用掩码去合**成，因此对于求解这个速度形式的话，也可以使用这种方法。

如果我们可以求解出来$$u_t^{target}(X_t|z)$$的话，就是需要知道如何将其合成起来即可。

#### 1.从$$p_t(x|z)$$到$$u_t^{target}(X_t|z)$$

由于我们有$$\dfrac{d}{dt}X_t=u_t(X_t|z)$$,则将$$X_t=p_t(x|z)$$带进去即为$$\dfrac{d}{dt}p_t(x|z)=u_t(p_t(x|z))$$,从而解微分方程即可

#### 2.从$$u_t^{target}(X_t|z)$$到$$u_t^{target}(X_t)$$

对于这个可以通过如下定理内容实现

- **定理1：**对于$$z\in R^d$$,使用$$u_t^{target}(.|z)$$表示一个条件速度场，如果对于从$$X_0\sim p_{init} \Rightarrow X_t\sim p_t(.|z)$$可以使用$$\dfrac{d}{dt}X_t=u_t^{target}(.|z)$$去描述，则有$$u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x|z) \frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)} \,\mathrm{d}z$$

可以使得$$X_0\sim p_{init} \Rightarrow X_t\sim p_t(x)$$对应的速度场为$$\dfrac{d}{dt}X_t=u_t^{target}(X_t)$$

- **定理2**：对于$$z\in R^d$$,使用$$u_t^{target}(.|z)$$表示一个条件速度场，如果对于从$$X_0\sim p_{init} \Rightarrow X_t\sim p_t(.|z)$$可以使用$$dX_t=u_t^{target}(X_t|z)dt+\sigma_t dW_t$$去描述,则有$$\mathrm{d}X_t = \left[ u_t^{\text{target}}(X_t) + \frac{\sigma_t^2}{2} \nabla \log p_t(X_t) \right] \mathrm{d}t + \sigma_t \,\mathrm{d}W_t$$，其中$$\nabla \log p_t(X_t)$$是**边缘得分函数**，$$\nabla \log p_t(x) = \int \nabla \log p_t(x|z) \frac{p_t(x|z) p_{\text{data}}(z)}{p_t(x)} \,\mathrm{d}z$$

**证明：**

我们只证明定理2，定理1实际上是定理2的在$$\sigma_t=0$$的特殊形式

引理(Fokker-Planck Equation):对于一个随机过程$$X_t=p_t(x)$$,它满足$$dX_t=u_t(X_t)dt+\sigma_tdW_t$$等价于满足

$$\partial_t p_t(x) = -\mathrm{div}(p_t u_t)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x)
\quad \text{for all } x \in \mathbb{R}^d, 0 \leq t \leq 1.$$

(引理的证明见附录)

对于定理1，则有$$\begin{align*}
\partial_t p_t(x) 
&{=} \partial_t \int p_t(x|z) p_{\text{data}}(z) \,\mathrm{d}z \\
&= \int \partial_t p_t(x|z) p_{\text{data}}(z) \,\mathrm{d}z \\
&{=} \int -\mathrm{div}\big(p_t(\cdot|z) u_t^{\text{target}}(\cdot|z)\big)(x) p_{\text{data}}(z) \,\mathrm{d}z \\
&{=} -\mathrm{div}\left( \int p_t(x|z) u_t^{\text{target}}(x|z) p_{\text{data}}(z) \,\mathrm{d}z \right) \\
&{=} -\mathrm{div}\left( p_t(x) \int u_t^{\text{target}}(x|z) \frac{p_t(x|z) p_{\text{data}}(z)}{p_t(x)} \,\mathrm{d}z \right)(x) \\
&{=} -\mathrm{div}\big(p_t u_t^{\text{target}}\big)(x),
\end{align*}$$

定理2则为

$$\begin{aligned}
\partial_t p_t(x) 
&{=} -\operatorname{div}(p_t u_t^{\text{target}})(x) \\
&{=} -\operatorname{div}(p_t u_t^{\text{target}})(x) - \frac{\sigma_t^2}{2} \Delta p_t(x) + \frac{\sigma_t^2}{2} \Delta p_t(x) \\
&{=} -\operatorname{div}(p_t u_t^{\text{target}})(x) - \operatorname{div}\left( \frac{\sigma_t^2}{2} \nabla p_t \right)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x) \\
&{=} -\operatorname{div}(p_t u_t^{\text{target}})(x) - \operatorname{div}\left( p_t \left[ \frac{\sigma_t^2}{2} \nabla \log p_t \right] \right)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x) \\
&{=} -\operatorname{div}\left( p_t \left[ u_t^{\text{target}} + \frac{\sigma_t^2}{2} \nabla \log p_t \right] \right)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x),
\end{aligned}$$

从而我们证明了这两个定理

有了这个定理，我们就可以将$$u_t^{target}(x|z)$$转换成$$u_t^{target}(x)$$。

## 损失函数

我们已经知道对于我们的模型的初始假设：更新时$$p_t(x|z)$$是如何从$$p_{init}$$到$$\delta_z$$的，衡量标准$$u_t^{target}$$，以及如何进行预测以及更新（执行ODE），现在需要做的就是寻找一个**合适的损失函数**。

## Flow Matching

对于flow model而言，我们的条件是$$X_0 \sim p_{\text{init}}, \quad \mathrm{d}X_t = u_t^\theta(X_t)\, \mathrm{d}t.$$

而我们希望有$$u_t^\theta \approx u_t^{target}$$,那么自然而然的，想要使用对于**速度场**的L2损失

（stackrel 为在=上面添加下标）

$$\mathcal{L}_{\mathrm{FM}}(\theta) = \mathbb{E}_{t \sim \mathrm{Unif}, x \sim p_t} \bigl[ \| u_t^\theta(x) - u_t^{\mathrm{target}}(x) \|^2 \bigr] \\
= \mathbb{E}_{t \sim \mathrm{Unif}, z \sim p_{\mathrm{data}}, x \sim p_t(\cdot \mid z)} \bigl[ \| u_t^\theta(x) - u_t^{\mathrm{target}}(x) \|^2 \bigr]$$

但是这样的损失函数不是特别容易计算，特别是对于$$u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x|z) \frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)} \,\mathrm{d}z$$这个东西，这里的积分由于z是个连续的东西，因此不容易计算。相应的，我们应该仍然考虑通过容易获得的$$u_t^{target}(x|z)$$去计算我们的损失函数。于是我们定义

$$L_{CFM}(\theta)=\mathbb{E}_{t \sim \mathrm{Unif}, z \sim p_{\mathrm{data}}, x \sim p_t(\cdot \mid z)} \bigl[ \| u_t^\theta(x) - u_t^{\mathrm{target}}(x|z) \|^2 \bigr]$$

但是我们由下述定理

**定理3（条件损失等价于边缘损失）**

$$L_{FM}(\theta)=L_{CFM}(\theta)+C$$,其中C为与参数无关的常数，即有$$\nabla_{\theta}L_{FM}(\theta)=\nabla_{\theta}L_{CFM}(\theta)$$,这说明了两种损失是等价的

**Proof:**

$$\begin{aligned}
\mathcal{L}_{\text{FM}}(\theta) 
&= \mathbb{E}_{\tau \sim \text{Unif},\, x \sim p_\epsilon} \left[ \| u_t^\theta(x) - u_t^{\text{target}}(x) \|^2 \right] \\
&= \mathbb{E}_{\tau \sim \text{Unif},\, x \sim p_\epsilon} \left[ \| u_t^\theta(x) \|^2 - 2\, u_t^\theta(x)^T u_t^{\text{target}}(x) + \| u_t^{\text{target}}(x) \|^2 \right] \\
&= \mathbb{E}_{\tau \sim \text{Unif},\, x \sim p_\epsilon} \left[ \| u_t^\theta(x) \|^2 \right] 2\underbrace{\mathbb{E}_{\tau \sim \text{Unif},\, x \sim p_\epsilon} \left[ u_t^\theta(x)^T u_t^{\text{target}}(x) \right]}_{\substack{= \displaystyle \int_0^1 \int_x p_\epsilon(x)\, u_t^\theta(x)^T u_t^{\text{target}}(x)\, dx\, d\tau \\ = \displaystyle \int_0^1 \int_z \int_x u_t^{\text{target}}(x|z)\, p_\epsilon(x|z)\, p_{\text{data}}(z)\, dx\, dz\, d\tau \\ = \displaystyle \mathbb{E}_{\tau \sim \text{Unif},\, z \sim p_{\text{data}},\, x \sim p_\epsilon(\cdot|z)} \left[ u_t^\theta(x)^T u_t^{\text{target}}(x|z) \right]}}-C_1 \\
&= \mathbb{E}_{\tau \sim \text{Unif},\, z \sim p_{\text{data}},\, x \sim p_\epsilon(\cdot|z)} \left[ \| u_t^\theta(x) \|^2 - 2\, u_t^\theta(x)^T u_t^{\text{target}}(x|z) + \| u_t^{\text{target}}(x|z) \|^2 - \| u_t^{\text{target}}(x|z) \|^2 \right] + C_1 \\
&= \mathcal{L}_{\text{CFM}}(\theta) + C_2 + C_1,
\end{aligned}$$

因此原本使用$$L_{FM}(\theta)$$实际上只需要使用$$L_{CFM}(\theta)$$就可以了

从而对于Flow matching的训练过程即为

```python
Step 1: initialize with a dataset z~pdata
Step 2:
    for each mini-batch of data:
        sample z from dataset
        sample t from [0,1]
        sample epsilon~N(0,Id)
        sample x~pt(x|z)
        Loss=|u(x)-utargert(x|z)|2
        update the parameters
```

## Score Matching

对于diffusion model而言，我们是通过$$\mathrm{d}X_t = \left[ u_t^{\text{target}}(X_t) + \frac{\sigma_t^2}{2} \nabla \log p_t(X_t) \right] \mathrm{d}t + \sigma_t \,\mathrm{d}W_t$$来更新函数的，但是同样的道理，我们无法直接去计算$$\nabla \log p_t(X_t) $$,但是我们可以通过一个**score network**得到$$s_t^\theta(x)$$,从而有

$$\mathcal{L}_{\mathrm{SM}}(\theta) 
= \mathbb{E}_{t \sim \mathrm{Unif}, z \sim p_{\mathrm{data}}, x \sim p_t(\cdot \mid z)} \bigl[ \| s_t^\theta(x) - \nabla \log p_t(x)\|^2 \bigr]$$

类似于上面的讨论，我们可以用同样的定理以及推导方式发现我们只需要去计算$$\mathcal{L}_{\mathrm{CSM}}(\theta) 
= \mathbb{E}_{t \sim \mathrm{Unif}, z \sim p_{\mathrm{data}}, x \sim p_t(\cdot \mid z)} \bigl[ \| s_t^\theta(x) - \nabla \log p_t(x|z)\|^2 \bigr]$$即可

综上所述，对于Score Matching的训练过程即为

```python
Step 1: initialize with a dataset z~pdata
Step 2:
    for each mini-batch of data:
        sample z from dataset
        sample t from [0,1]
        sample epsilon~N(0,Id)
        sample x~pt(x|z)
        Loss=|s(x)-partial log p(x|z)|2
        update the parameter
```

## 给定指令下的图片生成

以上的部分我们使用的数据集默认没有指令,即为$$p_t^\theta(x|\phi)$$,我们发现此时生成的图片随机性非常的大，不是我们想要的那些图片。我们需要添加一些的指示y，想要在$$p^\theta_t(x|y)$$生成的图片，我们称有指令的generative model为guided generative model.

guided generative model:我们的参数为

$$\begin{align*}
\textbf{Neural network:} & \quad u^\theta : \mathbb{R}^d \times \mathcal{Y} \times [0,1] \to \mathbb{R}^d, \; (x, y, t) \mapsto u_t^\theta(x|y) \\
\textbf{Fixed:} & \quad \sigma_t : [0,1] \to (0,\infty), \; t \mapsto \sigma_t
\end{align*}$$

而我们的模型为

$$\begin{align*}
\textbf{Initialization:} & \quad X_0 \sim p_{\text{init}} \\
\textbf{Simulation:} & \quad \mathrm{d}X_t = u^\theta_t(X_t|y)\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t \\
\textbf{Goal:} & \quad X_1 \sim p_{\text{data}}(\cdot|y)
\end{align*}$$

而我们需要加上指令，只需要在训练的时候将一个输出以及一条指令看成一个**pair**，他们是从$$(z,y)\sim p_{data}(z,y)$$中选取的，从而有损失函数变成了$$\mathcal{L}_{\text{CFM}}^{\text{guided}}(\theta) = \mathbb{E}_{(z,y)\sim p_{\text{data}}(z,y),\, t\sim\text{Unif}[0,1),\, x\sim p_t(\cdot\mid z)} \|u_t^\theta(x\mid y) - u_t^{\text{target}}(x\mid z)\|^2.$$

对于$$u_t^{target}(x|z)$$是在y的条件下进行**图片的提取**，然后我们的网络会结合y的指令，想办法得到理论上应该在y的条件下产生的图片差不多的图片。从而可以得到我们的结果。

### Classifier Free Guidance(CFG)

虽然如此，但是我们在实际的应用中发现，如果直接使用这个网络去实现的话，效果不是特别好。我们的策略是**奖励通过y的指令的图片，惩罚那些没有通过y的指令的图片**。

具体的过程为

$$\begin{align*}
u_t^{\text{target}}(x|y) &= a_t x + b_t \nabla \log p_t(x|y), \\
&= a_t x + b_t (\nabla \log p_t(x) + \nabla \log p_t(y|x)) = u_t^{\text{target}}(x) + b_t \nabla \log p_t(x|y).
\end{align*}$$

从而对于右侧的一项**添加一个guidance scale**，有$$\tilde{u}_t(x|y) = u_t^{\text{target}}(x) + w b_t \nabla \log p_t(y|x)$$

这一个权重的原因是因为我们知道$$\nabla log p_t(y|x)$$表示对于noised data的排除（因为我们的pt会比较的平均，导致整体的变化值不大，导致梯度不大)

于是我们的w就是让模型知道这个**对于y指令非常重要**，需要添加一个权重，此时我们再带回去，即有

$$\begin{align*}
\tilde{u}_t(x|y) 
&= u_t^{\text{target}}(x) + w_b t \nabla \log p_t(y|x) \\
&= u_t^{\text{target}}(x) + w_b t (\nabla \log p_t(x|y) - \nabla \log p_t(x)) \\
&= u_t^{\text{target}}(x) - (w_a t x + w_b t \nabla \log p_t(x)) + (w_a t x + w_b t \nabla \log p_t(x|y)) \\
&= (1 - w) u_t^{\text{target}}(x) + w u_t^{\text{target}}(x|y).
\end{align*}$$

此时对于Flow Model而言，在预测的时候，使用$$dX_t=\tilde{u_t}^\theta(X_t|y)dt$$去预测，同样的道理，对于Diffusion Model而言，我们也有$$\tilde{s}_t^\theta(x|y) = (1 - w) s_t^\theta(x|\varnothing) + w s_t^\theta(x|y)$$，预测的时候，使用$$\mathrm{d}X_t = \left[ \tilde{u}_t^\theta(X_t|y) + \frac{\sigma_t^2}{2} \tilde{s}_t^\theta(X_t|y) \right] \mathrm{d}t + \sigma_t \mathrm{d}W_t$$去预测

## Example:Gaussian probability path

  由于我们能想到的最简单的分布**就是高斯分布**，高斯分布作为$$p_t(x|z)$$的假设部分便显得非常的自然。接下来我们将会以$$p_t(x|z)=\mathcal{N}(\alpha_t z,\beta^2_t I_d)$$为假设，推导出$$u_t^{target}(x|z),\nabla log p_t(x|z)$$的具体形式，来实践以下以上的整个训练&预测的流程

由$$p_t(x|z)=\mathcal{N}(\alpha_t z,\beta^2_t I_d)$$,我们可以得到$$x=\alpha_t z+\beta_t \epsilon,\epsilon\in N(0,I_d)$$(obvious)

带入ODE中

$$\begin{aligned}
& \boxed{\alpha_t z + \beta_t x} = u_t^{\text{target}}(\alpha_t z + \beta_t x \mid z) \quad \text{for all } x, z \in \mathbb{R}^d \\
& \iff \alpha_t z + \beta_t \left( \frac{x - \alpha_t z}{\beta_t} \right) = u_t^{\text{target}}(x \mid z) \quad \text{for all } x, z \in \mathbb{R}^d \\
& \iff \left( \alpha_t - \frac{\beta_t}{\beta_t} \alpha_t \right) z + \frac{\beta_t}{\beta_t} x = u_t^{\text{target}}(x \mid z) \quad \text{for all } x, z \in \mathbb{R}^d
\end{aligned}$$

带入道$$\nabla log p_t(x|z)$$中，有$$\nabla \log p_t(x \mid z) = \nabla \log \mathcal{N}(x; \alpha_t z, \beta_t^2 I_d) = -\frac{x - \alpha_t z}{\beta_t^2}$$

这些即为我们的target的来源，在训练以及预测的时候直接带入即可。



### $$u_t^\theta(x)$$的网络结构

早期的diffusion Model借鉴了U-Nets的结构Encoder,decoder,midcoder的结构，通过多层的Encoder+CNN+Resets的结构，再通过一个比较小的midcoder，最后以和Encoder同样大小的Decoder把信息进行输出。

而在更加现代化的Diffusion Model中，我们常常使用一些特别的注意力机制去实现这些功能（具体的以我目前的水平还无法特别好的去理解，之后学完cs224n之后应该可以来完善这一块的内容）



而在为了将我们的指令变成便于计算机理解的“y”，对于主流的模型

stable diffusion: clip&t5-xxl等embedding的方式，同时使用MM-DiT的方式

Meta Gen Video:3 embeddings, cross&self attention

最后我们使用MNIST作为数据集，在google colab上使用T4跑了大概5min，得到一个可以生成数字的东西

---

  

以上就是对于flow&diffusion model的Generative AI的简介，包括初始的假设，到我们的评判标准，到训练以及预测的过程，可以说是套着边缘分布，使用条件分布去实现的一个非常美丽的模型。

但是对于$$u_t^\theta(x)$$的结构，以及模型的各种假设还没有进行比较细致的讨论，更多的内容之后再UCB CS294-158以及[MIT 6.S978](https://mit-6s978.github.io/?utm_source=chatgpt.com)去了解。



最后感谢MIT提供的如此优秀的课程，感谢所有的Mentor和TA做出来的这么美丽的课程，但是要是你们的作业的代码质量更高就好了(我的评价是史山)











