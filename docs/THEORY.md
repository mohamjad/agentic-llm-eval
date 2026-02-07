# Theoretical Foundations

## Overview

This document provides the mathematical and theoretical foundations underlying the agentic-llm-eval framework, including reinforcement learning algorithms, statistical methods, and optimization techniques.

## 1. Reinforcement Learning Framework

### 1.1 Policy Gradient Methods

We use **Proximal Policy Optimization (PPO)** for training agent parameters. PPO is a policy gradient method that optimizes a stochastic policy $\pi_\theta(a|s)$ parameterized by $\theta$.

#### Policy Objective

The standard policy gradient objective is:

$$J(\theta) = \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta} [\hat{A}(s, a)]$$

where $\hat{A}(s, a)$ is an estimate of the advantage function.

#### PPO Clipped Objective

PPO uses a clipped surrogate objective to prevent large policy updates:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the importance sampling ratio
- $\epsilon$ is the clipping parameter (typically 0.2)
- $\hat{A}_t$ is the advantage estimate

#### Value Function

We estimate the value function $V^\pi(s)$ using a separate neural network:

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t r_{t+1} \middle| s_0 = s \right]$$

The value loss is:

$$L^{VF}(\theta) = \mathbb{E}_t \left[ (V_\theta(s_t) - \hat{V}_t)^2 \right]$$

#### Total Loss

The total PPO loss combines policy, value, and entropy terms:

$$L^{PPO}(\theta) = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 H[\pi_\theta](\cdot|s_t)$$

where:
- $c_1$ is the value coefficient (typically 0.5)
- $c_2$ is the entropy coefficient (typically 0.01)
- $H[\pi_\theta]$ is the entropy bonus for exploration

### 1.2 Advantage Estimation

We use **Generalized Advantage Estimation (GAE)** to compute advantages:

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$$

where:
- $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error
- $\lambda \in [0, 1]$ is the GAE parameter (typically 0.95)
- $\gamma$ is the discount factor (typically 0.99)

### 1.3 Neural Network Architecture

#### Policy Network

The policy network $\pi_\theta(a|s)$ is a multi-layer perceptron:

$$\begin{align}
h_1 &= \text{ReLU}(W_1 s + b_1) \\
h_2 &= \text{ReLU}(W_2 h_1 + b_2) \\
\mu &= W_\mu h_2 + b_\mu \\
\sigma &= \exp(\text{clip}(\log\sigma_0, -2, 2))
\end{align}$$

The action is sampled from a Gaussian distribution:

$$a \sim \mathcal{N}(\mu, \sigma^2)$$

#### Value Network

The value network $V_\theta(s)$ has a similar architecture:

$$\begin{align}
h_1 &= \text{ReLU}(W_1 s + b_1) \\
h_2 &= \text{ReLU}(W_2 h_1 + b_2) \\
V(s) &= W_v h_2 + b_v
\end{align}$$

## 2. Bayesian Optimization

### 2.1 Gaussian Process Regression

We model the performance landscape $f(\mathbf{x})$ as a Gaussian Process:

$$f(\mathbf{x}) \sim \mathcal{GP}(\mu(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

where:
- $\mu(\mathbf{x})$ is the mean function (typically zero)
- $k(\mathbf{x}, \mathbf{x}')$ is the covariance kernel function

#### RBF Kernel

We use the Radial Basis Function (RBF) kernel:

$$k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{1}{2}\sum_{d=1}^D \frac{(x_d - x'_d)^2}{\ell_d^2}\right)$$

where:
- $\sigma_f^2$ is the signal variance
- $\ell_d$ are length scales for each dimension

### 2.2 Acquisition Functions

#### Expected Improvement (EI)

The Expected Improvement acquisition function:

$$\text{EI}(\mathbf{x}) = \mathbb{E}[\max(0, f(\mathbf{x}) - f(\mathbf{x}^+))]$$

where $f(\mathbf{x}^+)$ is the best observed value.

For a GP, this has a closed form:

$$\text{EI}(\mathbf{x}) = \sigma(\mathbf{x})[\Phi(Z) + Z\phi(Z)]$$

where:
- $Z = \frac{\mu(\mathbf{x}) - f(\mathbf{x}^+)}{\sigma(\mathbf{x})}$
- $\Phi$ and $\phi$ are the CDF and PDF of the standard normal distribution

## 3. Statistical Analysis

### 3.1 Confidence Intervals

For a sample mean $\bar{X}$ with sample size $n$ and sample standard deviation $s$, the $(1-\alpha)$ confidence interval is:

$$\bar{X} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}$$

where $t_{\alpha/2, n-1}$ is the critical value from the t-distribution.

### 3.2 Hypothesis Testing

#### Independent Samples t-test

For comparing two agents with samples $X_1$ and $X_2$:

$$t = \frac{\bar{X}_1 - \bar{X}_2}{s_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$$

where $s_p$ is the pooled standard deviation:

$$s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$$

### 3.3 Effect Size

#### Cohen's d

Cohen's d measures the standardized difference between two means:

$$d = \frac{\bar{X}_1 - \bar{X}_2}{s_p}$$

Interpretation:
- $|d| < 0.2$: Negligible
- $0.2 \leq |d| < 0.5$: Small
- $0.5 \leq |d| < 0.8$: Medium
- $|d| \geq 0.8$: Large

### 3.4 Bootstrap Sampling

Bootstrap confidence intervals use resampling:

1. Draw $B$ bootstrap samples with replacement
2. Compute statistic $\theta^*_b$ for each sample
3. Use percentiles of $\{\theta^*_b\}$ as confidence bounds

## 4. Semantic Similarity

### 4.1 Embedding Space

We use transformer-based embeddings to map text to a high-dimensional vector space:

$$\mathbf{e} = \text{Transformer}(\text{text}) \in \mathbb{R}^d$$

where $d$ is the embedding dimension (typically 384 for MiniLM).

### 4.2 Cosine Similarity

Semantic similarity is measured using cosine similarity:

$$\text{sim}(\mathbf{e}_1, \mathbf{e}_2) = \frac{\mathbf{e}_1 \cdot \mathbf{e}_2}{||\mathbf{e}_1|| \cdot ||\mathbf{e}_2||}$$

This measures the cosine of the angle between vectors, ranging from -1 to 1.

### 4.3 Normalized Similarity Score

We convert cosine similarity to a [0, 1] range:

$$\text{score} = \frac{\text{sim} + 1}{2}$$

## 5. Multi-Objective Optimization

### 5.1 Weighted Sum Method

We combine multiple metrics using weighted sums:

$$\text{score} = \sum_{i=1}^n w_i \cdot m_i$$

where:
- $w_i$ are weights ($\sum w_i = 1$)
- $m_i$ are normalized metric values

### 5.2 Pareto Optimality

A parameter set $\mathbf{x}^*$ is Pareto optimal if there exists no $\mathbf{x}$ such that:

$$m_i(\mathbf{x}) \geq m_i(\mathbf{x}^*) \quad \forall i$$

and

$$m_j(\mathbf{x}) > m_j(\mathbf{x}^*) \quad \text{for some } j$$

## 6. Convergence Analysis

### 6.1 Policy Gradient Convergence

Under certain conditions, policy gradient methods converge to a local optimum:

$$\lim_{t \to \infty} \nabla_\theta J(\theta_t) = 0$$

The convergence rate depends on the learning rate and the smoothness of the objective.

### 6.2 Bayesian Optimization Convergence

Bayesian optimization converges to the global optimum under certain conditions on the kernel and acquisition function. The convergence rate is typically $O(n^{-1/2})$ for certain kernels.

## 7. Computational Complexity

### 7.1 Neural Network Forward Pass

- Policy network: $O(d_h^2)$ where $d_h$ is hidden dimension
- Value network: $O(d_h^2)$
- Total per step: $O(d_h^2)$

### 7.2 PPO Update

- Forward pass: $O(B \cdot d_h^2)$ where $B$ is batch size
- Backward pass: $O(B \cdot d_h^2)$
- Total per update: $O(B \cdot d_h^2)$

### 7.3 Bayesian Optimization

- GP inference: $O(n^3)$ where $n$ is number of observations
- Acquisition optimization: $O(n^2)$ per evaluation
- Total per iteration: $O(n^3)$

## References

1. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347
2. Williams, R. J. (1992). "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." Machine Learning, 8(3-4), 229-256.
3. Rasmussen, C. E., & Williams, C. K. I. (2006). "Gaussian Processes for Machine Learning." MIT Press.
4. Jones, D. R., Schonlau, M., & Welch, W. J. (1998). "Efficient Global Optimization of Expensive Black-Box Functions." Journal of Global Optimization, 13(4), 455-492.
5. Cohen, J. (1988). "Statistical Power Analysis for the Behavioral Sciences." Routledge.
