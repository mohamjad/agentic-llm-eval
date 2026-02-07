# Theoretical Foundations

Mathematical and theoretical foundations underlying the agentic-llm-eval framework, including reinforcement learning algorithms, statistical methods, and optimization techniques.

We use Proximal Policy Optimization (PPO) for training agent parameters. PPO is a policy gradient method that optimizes a stochastic policy π_θ(a|s) parameterized by θ. The standard policy gradient objective is J(θ) = E_{s ~ ρ^π, a ~ π_θ} [Â(s, a)] where Â(s, a) is an estimate of the advantage function.

PPO uses a clipped surrogate objective to prevent large policy updates: L^CLIP(θ) = E_t [min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)] where r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t) is the importance sampling ratio, ε is the clipping parameter typically 0.2, and Â_t is the advantage estimate.

We estimate the value function V^π(s) using a separate neural network: V^π(s) = E_π [Σ_{t=0}^∞ γ^t r_{t+1} | s_0 = s]. The value loss is L^VF(θ) = E_t [(V_θ(s_t) - V̂_t)^2].

The total PPO loss combines policy, value, and entropy terms: L^PPO(θ) = L^CLIP(θ) - c_1 L^VF(θ) + c_2 H[π_θ](·|s_t) where c_1 is the value coefficient typically 0.5, c_2 is the entropy coefficient typically 0.01, and H[π_θ] is the entropy bonus for exploration.

We use Generalized Advantage Estimation (GAE) to compute advantages: Â_t = δ_t + (γλ)δ_{t+1} + (γλ)^2 δ_{t+2} + ... where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD error, λ ∈ [0, 1] is the GAE parameter typically 0.95, and γ is the discount factor typically 0.99.

The policy network π_θ(a|s) is a multi-layer perceptron with h_1 = ReLU(W_1 s + b_1), h_2 = ReLU(W_2 h_1 + b_2), μ = W_μ h_2 + b_μ, and σ = exp(clip(log σ_0, -2, 2)). The action is sampled from a Gaussian distribution a ~ N(μ, σ^2).

The value network V_θ(s) has a similar architecture with h_1 = ReLU(W_1 s + b_1), h_2 = ReLU(W_2 h_1 + b_2), and V(s) = W_v h_2 + b_v.

For Bayesian optimization, we model the performance landscape f(x) as a Gaussian Process: f(x) ~ GP(μ(x), k(x, x')) where μ(x) is the mean function typically zero, and k(x, x') is the covariance kernel function.

We use the Radial Basis Function (RBF) kernel: k(x, x') = σ_f^2 exp(-(1/2)Σ_{d=1}^D (x_d - x'_d)^2 / ℓ_d^2) where σ_f^2 is the signal variance and ℓ_d are length scales for each dimension.

The Expected Improvement (EI) acquisition function is EI(x) = E[max(0, f(x) - f(x^+))] where f(x^+) is the best observed value. For a GP this has closed form: EI(x) = σ(x)[Φ(Z) + Zφ(Z)] where Z = (μ(x) - f(x^+)) / σ(x), and Φ and φ are the CDF and PDF of the standard normal distribution.

For statistical analysis, confidence intervals for a sample mean X̄ with sample size n and sample standard deviation s use the (1-α) confidence interval: X̄ ± t_{α/2, n-1} s / √n where t_{α/2, n-1} is the critical value from the t-distribution.

Independent samples t-test compares two agents with samples X_1 and X_2 using t = (X̄_1 - X̄_2) / (s_p √(1/n_1 + 1/n_2)) where s_p is the pooled standard deviation: s_p = √(((n_1-1)s_1^2 + (n_2-1)s_2^2) / (n_1 + n_2 - 2)).

Cohen's d measures the standardized difference between two means: d = (X̄_1 - X̄_2) / s_p. Interpretation considers |d| < 0.2 negligible, 0.2 ≤ |d| < 0.5 small, 0.5 ≤ |d| < 0.8 medium, and |d| ≥ 0.8 large.

Bootstrap confidence intervals use resampling by drawing B bootstrap samples with replacement, computing statistic θ*_b for each sample, and using percentiles of {θ*_b} as confidence bounds.

Semantic similarity uses transformer-based embeddings to map text to high-dimensional vector space: e = Transformer(text) ∈ R^d where d is the embedding dimension typically 384 for MiniLM. Cosine similarity measures semantic similarity: sim(e_1, e_2) = (e_1 · e_2) / (||e_1|| · ||e_2||), ranging from -1 to 1. We convert to [0, 1] range using score = (sim + 1) / 2.

Multi-objective optimization combines multiple metrics using weighted sums: score = Σ_{i=1}^n w_i · m_i where w_i are weights summing to 1, and m_i are normalized metric values. A parameter set x* is Pareto optimal if there exists no x such that m_i(x) ≥ m_i(x*) for all i and m_j(x) > m_j(x*) for some j.

Policy gradient methods converge to a local optimum under certain conditions: lim_{t→∞} ∇_θ J(θ_t) = 0. The convergence rate depends on learning rate and objective smoothness.

Bayesian optimization converges to the global optimum under certain conditions on the kernel and acquisition function. The convergence rate is typically O(n^{-1/2}) for certain kernels.

Computational complexity for neural network forward pass is O(d_h^2) where d_h is hidden dimension. PPO update is O(B · d_h^2) where B is batch size. Bayesian optimization GP inference is O(n^3) where n is number of observations, acquisition optimization is O(n^2) per evaluation, totaling O(n^3) per iteration.

References include Schulman et al. 2017 on Proximal Policy Optimization Algorithms, Williams 1992 on Simple Statistical Gradient-Following Algorithms, Rasmussen and Williams 2006 on Gaussian Processes for Machine Learning, Jones et al. 1998 on Efficient Global Optimization, and Cohen 1988 on Statistical Power Analysis.
