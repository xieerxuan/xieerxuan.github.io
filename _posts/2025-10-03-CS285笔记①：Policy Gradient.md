---
layout: post
title: "CS285 S1: Policy Gradient"
author: xex
date: 2025-10-03
categories: Notes
tags: reinforcement-learning
usemathjax: true
---


## 1. Recap the Goal of RL

强化学习的目标是最大化期望回报：

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_t r(s_t, a_t) \right] 
$$

因此，强化学习的参数学习目标为：

$$ 
\theta^* = \arg\max_\theta J(\theta )= \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_t r(s_t, a_t) \right] 
$$

其中

$$ p(\tau) = p(s_1,a_1,s_2,a_2,...,s_T,a_T)= p(s_1)\prod_{t=1}^{T} p(s_{t+1} \mid s_t, a_t)$$ 

是某条 trajectory 出现的概率（密度）。


<div class="box-info" markdown="1">
<div class="title"> INSIGHTS </div>
看起来，我们只需要对 $$J(\theta)$$ 进行梯度上升，就可以优化参数 $$\theta$$。然而，由于 $$ p(\tau) $$ 中存在 $$ p(s_1) $$ 和 $$p(s_{t+1}\mid s_t,a_t)$$ 这些与 $\theta$ 有关的未知项，所以我们无法直接求得 $$\nabla_\theta J(\theta)$$ 。那么我们能否绕开对环境的建模，即在未知 $$ p(s_1) $$和$$p(s_{t+1}\mid s_t,a_t)$$ 的情况下，对 $$J(\theta)$$ 进行梯度上升呢？这就是下一节中REINFORCE算法所做的工作。
</div>


## 2. REINFORCE Algorithm
设轨迹 $\tau \sim p_\theta(\tau)$，则

$$
J(\theta)=E_{\tau\sim\pi(t)}\left[R(\tau)\right]=\int_{\theta}p_{\theta}(\tau)R(\tau)d\tau
$$

其中，$R(\tau)$ 是某一轨迹 $\tau$ 下的奖励和 $ \sum_t r(s_t, a_t)$ （当然也可以加上衰减因子 $\gamma$）。我们尝试对目标函数直接求梯度：

$$ 
\begin{align*}
\nabla_{\theta}J(\theta)&=\int\nabla_{\theta}p_{\theta}(\tau)R(\tau)d\tau\\\\
&=\int p_{\theta}(\tau)\nabla_{\theta}\log p_{\theta}(\tau)R(\tau)d\tau\\\\
&=E_{\tau\sim p_\theta(\tau)}\left[\nabla_{\theta}\log p_{\theta}(\tau)R(\tau)\right]\\\\
&=E_{\tau\sim p_\theta(\tau)}\left[\left( \sum_{t=1}^{T}\nabla_\theta \log \pi_\theta (a_t^i\mid s_t^i)\right)
\left( \sum_{t=1}^{T}r(a_t^i\mid s_t^i)\right)\right]
\end{align*} 
$$

<div class="box-tip" markdown="1">
<div class="title"> NOTES </div>
上式第二个等号：

$$\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} = p_\theta(\tau) \nabla_\theta [\log p_\theta(\tau)]$$

第四个等号：

$$
\begin{align*}
\log p_\theta(\tau) &= \log p(s_1) + \sum_{t=1}^{T}\log \pi_\theta(a_t \mid s_t) + \log p (s_{t+1}\mid s_t,a_t)\\
\nabla_\theta\left[\log p_\theta(\tau)\right] &= \nabla_\theta \sum_{t=1}{T}\log \pi_\theta(a_t \mid s_t)
\end{align*}
$$

</div>


因此，我们得到采样下的近似梯度

$$ \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t^i|s_t^i) \right) \left( \sum_{t=1}^{T} r(s_t^i, a_t^i) \right) $$

对策略参数 $\theta$ 进行梯度上升：

$$
 \theta \leftarrow \theta + \alpha \nabla_\theta J(\theta) 
 $$

我们就得到了 REINFORCE 算法，步骤如下：
<div class="box-warning" markdown="1">
<div class="title"> REINFORCE 算法 </div>
1. 运行当前策略: 从 $\pi_\theta(a_t\mid s_t)$ 采样轨迹 $\tau_i$
2. 计算策略梯度估计：
   
   $$ \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t^i|s_t^i) \right) \left( \sum_{t=1}^{T} r(s_t^i, a_t^i) \right) $$

3. 参数更新：$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$
</div>

## 3. Policy Gradient vs MLE

$$ 
\begin{align*}
  \text{PG:} \quad \nabla_\theta J(\theta) &\approx \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t^i|s_t^i) \right) \left( \sum_{t=1}^{T} r(s_t^i, a_t^i) \right)\\\\

  \text{MLE:} \quad \nabla_\theta J(\theta) &\approx \frac{1}{N} \sum_{i=1}^{N} \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t^i|s_t^i) \right)

\end{align*}
$$

**最大似然估计**直接增加专家动作被产生的概率；**策略梯度**奖励由 $\pi_\theta$ 产生，以奖励作为加权自我调节 $\pi_\theta$ 的参数，增加总奖励值为正的动作产生的概率。对 PG 算法来说：
- 当 $\sum_t r(s_t, a_t) > 0$ 时：增加该动作序列的概率
- 当 $\sum_t r(s_t, a_t) < 0$ 时：减小该动作序列的概率

## 4. Prob. & Sol. 1: High Variance | Causality and Baseline

### 解决方案1：因果性(Causality)
利用因果关系，只考虑当前时刻之后的奖励：

$$ \nabla_\theta J(\theta) = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t^i|s_t^i) \left( \sum_{t'=t}^{T} r(s_{t'}^i, a_{t'}^i) \right) $$

其中 $\sum_{t'=t}^{T} r(s_{t'}^i, a_{t'}^i) $ 也被称为“reward to go”。

### 解决方案2：基线(Baseline)
引入基线减少方差：

$$ \nabla_\theta J(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \log p_\theta(\tau) [r(\tau) - b] $$

基线可以选择为：$b = \frac{1}{N} \sum_{i=1}^{N} r(\tau)$

**无偏性证明**：
$$ \mathbb{E}[\nabla_\theta \log p_\theta(\tau) b] = \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) b  d\tau = b \nabla_\theta \int p_\theta(\tau) d\tau = b \nabla_\theta 1 = 0 $$

## 5. Prob. & Sol. 2: Sampling Efficiency | Off-line Policy Gradient

### 解决方案：重要性采样(Importance Sampling)
由重要性采样定理：

$$ \mathbb{E}_{x \sim p(x)}[f(x)] = \int p(x) f(x) dx = \int q(x) \frac{p(x)}{q(x)} f(x) dx = \mathbb{E}_{x \sim q(x)} \left[ \frac{p(x)}{q(x)} f(x) \right] $$

我们可以得到 off-policy 版本的策略梯度：

$$ \nabla_{\theta'} J(\theta') = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \frac{p_{\theta'}(\tau)}{p_\theta(\tau)} \nabla_{\theta'} \log p_{\theta'}(\tau) r(\tau) \right] $$

考虑因果关系后的完整形式：
<div class="box-warning" markdown="1">
<div class="title"> Off-policy PG </div>

$$
\begin{aligned}
\nabla_{\theta'} J(\theta') = &\mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \left( \prod_{t=1}^{T} \frac{\pi_{\theta'}(a_t|s_t)}{\pi_\theta(a_t|s_t)} \right) \left( \sum_{t=1}^{T} \nabla_{\theta'} \log \pi_{\theta'}(a_t|s_t) \right) \left( \sum_{t=1}^{T} r(s_t, a_t) \right) \right] \\\\
\overset{\text{Causality}}{=} &\mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_{t=1}^{T} \nabla_{\theta'} \log \pi_{\theta'}(a_t|s_t) \left( \prod_{t'=1}^{t} \frac{\pi_{\theta'}(a_{t'}|s_{t'})}{\pi_\theta(a_{t'}|s_{t'})} \right) \left( \sum_{t'=t}^{T} r(s_{t'}, a_{t'}) \left( \prod_{t''=t}^{t'} \frac{\pi_{\theta'}(a_{t''}|s_{t''})}{\pi_\theta(a_{t''}|s_{t''})} \right) \right) \right]
\end{aligned}
$$
</div>