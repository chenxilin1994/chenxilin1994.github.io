# EM算法

## EM算法原理详解（含公式推导）



### 1. 核心思想与问题定义  
EM（Expectation-Maximization）算法是用于**含隐变量概率模型**参数估计的迭代方法，解决以下问题：  
- **观测数据**：$X = \{x_1, ..., x_N\}$  
- **隐变量**：$Z = \{z_1, ..., z_N\}$（未观测的类别标签或潜在变量）  
- **目标**：极大化对数似然函数 $\ln p(X \mid \theta)$，但因隐变量存在导致直接优化困难。  

### 2. 算法步骤的数学描述  
**输入**：观测数据 $X$，联合分布 $p(X, Z \mid \theta)$，初始参数 $\theta^{(0)}$  
**输出**：最优参数 $\theta^*$，使得 $\ln p(X \mid \theta)$ 最大  

**步骤1：E步（期望步）**  
基于当前参数 $\theta^{(t)}$，计算隐变量后验分布的期望：  
$$
Q(\theta \mid \theta^{(t)}) = \mathbb{E}_{Z \mid X, \theta^{(t)}} \left[ \ln p(X, Z \mid \theta) \right]
$$  
具体计算为：  
$$
Q(\theta \mid \theta^{(t)}) = \sum_{Z} p(Z \mid X, \theta^{(t)}) \ln p(X, Z \mid \theta)
$$  

**步骤2：M步（最大化步）**  
更新参数以最大化期望：  
$$
\theta^{(t+1)} = \arg\max_{\theta} Q(\theta \mid \theta^{(t)})
$$  

**步骤3：收敛判定**  
重复E步和M步，直到满足 $|\ln p(X \mid \theta^{(t+1)}) - \ln p(X \mid \theta^{(t)})| < \epsilon$ 或达到最大迭代次数。



### 3. 关键公式推导  
**对数似然分解**：  
$$
\ln p(X \mid \theta) = \underbrace{Q(\theta \mid \theta^{(t)})}_{\text{期望项}} - \underbrace{D_{\text{KL}}\left( p(Z \mid X, \theta^{(t)}) \parallel p(Z \mid X, \theta) \right)
$$  
其中 $D_{\text{KL}}$ 为KL散度，非负性保证似然函数单调递增。

**Q函数表达式（以GMM为例）**：  
对于高斯混合模型，隐变量 $Z$ 表示样本所属簇，Q函数展开为：  
$$
Q(\theta \mid \theta^{(t)}) = \sum_{n=1}^N \sum_{k=1}^K \gamma(z_{nk}) \left[ \ln \pi_k + \ln \mathcal{N}(x_n \mid \mu_k, \Sigma_k) \right]
$$  
其中责任值 $\gamma(z_{nk}) = p(z_{nk}=1 \mid x_n, \theta^{(t)})$。



### 4. EM算法收敛性证明  
1. **单调性**：  
   - 每次迭代中，$\ln p(X \mid \theta^{(t+1)}) \geq \ln p(X \mid \theta^{(t)})$  
   - 由KL散度非负性及M步的极大化保证。  
2. **有界性**：若似然函数有上界，则必收敛到局部最优。  



## Python深度实践指南（扩展版）



### 1. 基础实现：手动编写EM算法  
**问题设定**：估计两个高斯分布的混合参数（均值、方差、权重）  
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 生成混合高斯数据
np.random.seed(0)
n_samples = 1000
mu_true = [1.5, 4.5]
sigma_true = [0.6, 0.8]
weights_true = [0.4, 0.6]
samples = np.concatenate([
    np.random.normal(mu_true[0], sigma_true[0], int(n_samples * weights_true[0])),
    np.random.normal(mu_true[1], sigma_true[1], int(n_samples * weights_true[1]))
])

# EM算法实现
def em_gmm(data, n_components=2, max_iter=100, tol=1e-6):
    # 初始化参数
    mu = np.random.choice(data, n_components)
    sigma = np.array([np.std(data)] * n_components)
    pi = np.ones(n_components) / n_components
    
    log_likelihood_history = []
    
    for _ in range(max_iter):
        # E步：计算责任值
        gamma = np.zeros((len(data), n_components))
        for k in range(n_components):
            gamma[:, k] = pi[k] * norm.pdf(data, mu[k], sigma[k])
        gamma /= gamma.sum(axis=1, keepdims=True)
        
        # M步：更新参数
        Nk = gamma.sum(axis=0)
        pi = Nk / len(data)
        mu = (gamma.T @ data) / Nk
        sigma = np.sqrt((gamma.T @ (data[:, None] - mu)**2) / Nk)
        
        # 计算对数似然
        log_likelihood = np.sum(np.log(np.sum([pi[k] * norm.pdf(data, mu[k], sigma[k]) 
                                             for k in range(n_components)], axis=0)))
        log_likelihood_history.append(log_likelihood)
        
        # 收敛判断
        if len(log_likelihood_history) > 1 and abs(log_likelihood_history[-1] - log_likelihood_history[-2]) < tol:
            break
    
    return mu, sigma, pi, log_likelihood_history

# 运行EM算法
mu_est, sigma_est, pi_est, llh = em_gmm(samples)
print(f"Estimated mu: {mu_est}, sigma: {sigma_est}, pi: {pi_est}")

# 可视化收敛过程
plt.plot(llh)
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")
plt.title("EM Algorithm Convergence")
plt.show()
```

### 2. 进阶应用：含缺失数据的参数估计  
**场景**：数据部分特征缺失，使用EM进行填充与参数估计  
```python
# 生成含缺失数据
data_complete = np.random.multivariate_normal(mean=[2, 5], cov=[[1, 0.8], [0.8, 1]], size=100)
missing_mask = np.random.rand(*data_complete.shape) < 0.3
data_missing = np.where(missing_mask, np.nan, data_complete)

# EM算法处理缺失数据
def em_with_missing(data, n_components=1, max_iter=100):
    # 初始化参数
    mu = np.nanmean(data, axis=0)
    cov = np.cov(data, rowvar=False, bias=True)
    
    for _ in range(max_iter):
        # E步：计算缺失值的期望
        data_imputed = data.copy()
        for i in range(len(data)):
            nan_indices = np.isnan(data[i])
            if nan_indices.any():
                observed = ~nan_indices
                mu_cond = mu[nan_indices] + cov[nan_indices][:, observed] @ np.linalg.inv(cov[observed][:, observed]) @ (data[i, observed] - mu[observed])
                data_imputed[i, nan_indices] = mu_cond
        
        # M步：更新参数
        mu = np.mean(data_imputed, axis=0)
        cov = np.cov(data_imputed.T, bias=True)
    
    return mu, cov

mu_est, cov_est = em_with_missing(data_missing)
print(f"Estimated mu: {mu_est}\nCovariance:\n{cov_est}")
```

### 3. 对比实验：EM vs 梯度下降  
```python
from scipy.optimize import minimize

# 定义负对数似然函数
def neg_log_likelihood(params, data):
    mu1, sigma1, mu2, sigma2, pi = params
    log_lik = np.log(pi * norm.pdf(data, mu1, sigma1) + (1 - pi) * norm.pdf(data, mu2, sigma2))
    return -np.sum(log_lik)

# 梯度下降优化
initial_guess = [1.0, 1.0, 4.0, 1.0, 0.5]
result = minimize(neg_log_likelihood, initial_guess, args=(samples,), method='L-BFGS-B')
print("Gradient Descent Estimates:", result.x)

# 对比EM结果
print("EM Estimates:", np.concatenate([mu_est, sigma_est, pi_est]))
```



## 数学补充证明

### 1. EM算法与Jensen不等式  
对于任意分布 $q(Z)$，有：  
$$
\ln p(X \mid \theta) \geq \mathbb{E}_q[\ln p(X, Z \mid \theta)] - \mathbb{E}_q[\ln q(Z)]
$$  
当且仅当 $q(Z) = p(Z \mid X, \theta)$ 时等号成立，此时下界紧贴原似然函数。

### 2. 混合模型的Q函数全局最优性  
在指数族分布假设下，M步的解析解存在且唯一。例如高斯分布的均值和协方差更新公式可严格推导。



## 总结与扩展方向  
1. **理论扩展**：  
   - 推导EM在隐马尔可夫模型（HMM）中的应用  
   - 研究变分EM（Variational EM）与随机EM（Stochastic EM）  

2. **计算优化**：  
   - 使用GPU加速大规模EM计算  
   - 在线EM处理流式数据  

3. **应用场景**：  
   - 推荐系统中的协同过滤（含缺失评分）  
   - 生物信息学的基因表达数据分析  
