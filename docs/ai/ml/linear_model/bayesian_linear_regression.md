

# 贝叶斯线性回归


## 一、贝叶斯线性回归理论基础

### 1. 核心思想与模型设定
贝叶斯线性回归（Bayesian Linear Regression）通过引入参数的**概率分布**，将传统线性回归扩展为概率框架下的推断方法。其核心优势在于：
- **量化不确定性**：提供参数和预测的完整概率分布
- **引入先验知识**：通过先验分布融合领域知识
- **避免过拟合**：自然实现正则化

#### 模型设定
假设观测数据满足：
$$
y = X\beta + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$
其中：
- $X \in \mathbb{R}^{n \times m}$：特征矩阵
- $\beta \in \mathbb{R}^{m}$：回归系数
- $\sigma^2$：噪声方差



### 2. 贝叶斯推断框架

#### 先验分布
假设参数服从共轭先验（Conjugate Prior）：
$$
\beta \sim \mathcal{N}(\mu_0, \Sigma_0)
$$
通常选择无信息先验：
$$
\mu_0 = 0, \quad \Sigma_0 = \lambda^{-1} I \ (\lambda \to 0)
$$

#### 似然函数
$$
p(y|X, \beta) = \mathcal{N}(X\beta, \sigma^2 I)
$$

#### 后验分布
根据贝叶斯定理：
$$
p(\beta|y,X) \propto p(y|X,\beta)p(\beta)
$$
对于共轭先验，后验分布仍为正态分布：
$$
\beta|y,X \sim \mathcal{N}(\mu_n, \Sigma_n)
$$
其中：
$$
\Sigma_n^{-1} = \frac{1}{\sigma^2} X^T X + \Sigma_0^{-1}
$$
$$
\mu_n = \Sigma_n \left( \frac{1}{\sigma^2} X^T y + \Sigma_0^{-1} \mu_0 \right)
$$



### 3. 预测分布
对新样本 $x_*$ 的预测分布：
$$
p(y_*|x_*, X, y) = \mathcal{N}(x_*^T \mu_n, \ x_*^T \Sigma_n x_* + \sigma^2)
$$



## 二、Python第三方库实践

### 1. 使用PyMC3进行MCMC采样

#### 环境准备
```python
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
np.random.seed(42)
X = np.linspace(0, 10, 100)
true_slope = 2.5
y = true_slope * X + np.random.normal(0, 3, 100)
```

#### 构建贝叶斯模型
```python
with pm.Model() as linear_model:
    # 先验分布
    slope = pm.Normal('slope', mu=0, sigma=10)
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    # 似然函数
    mu = slope * X + intercept
    likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
    
    # MCMC采样
    trace = pm.sample(2000, tune=1000, chains=4)
```

#### 结果分析与可视化
```python
pm.plot_trace(trace)
plt.show()

print(pm.summary(trace))
# 输出示例：
#          mean    sd   hdi_3%  hdi_97%
# slope    2.48  0.05    2.39    2.57
# intercept 0.12  0.28   -0.37    0.61
# sigma    2.92  0.21    2.55    3.30

# 后验预测检查
ppc = pm.sample_posterior_predictive(trace, model=linear_model)
plt.scatter(X, y, alpha=0.3)
for i in range(100):
    plt.plot(X, ppc['y'][i], 'gray', alpha=0.1)
plt.plot(X, true_slope*X, 'r--', label='True Line')
plt.legend()
plt.show()
```



### 2. 使用Scikit-Learn贝叶斯岭回归
```python
from sklearn.linear_model import BayesianRidge

# 模型训练
model = BayesianRidge(compute_score=True)
model.fit(X.reshape(-1,1), y)

# 获取参数分布
print(f"斜率均值: {model.coef_[0]:.2f} ± {model.sigma_[0][0]**0.5:.2f}")
print(f"截距均值: {model.intercept_:.2f} ± {model.sigma_[1][1]**0.5:.2f}")

# 可视化不确定性
X_test = np.linspace(0, 10, 100).reshape(-1,1)
y_mean, y_std = model.predict(X_test, return_std=True)

plt.fill_between(X_test.ravel(), 
                 y_mean - 2*y_std, 
                 y_mean + 2*y_std, 
                 alpha=0.3, 
                 label='95% CI')
plt.scatter(X, y, alpha=0.3)
plt.plot(X_test, y_mean, 'r', label='预测均值')
plt.legend()
plt.show()
```



## 三、手动实现贝叶斯线性回归

### 1. 共轭先验下的解析解

#### 公式实现
假设已知噪声方差 $\sigma^2 = 1$，先验参数 $\mu_0 = 0$, $\Sigma_0 = 10^2 I$：
```python
class BayesianLinearRegression:
    def __init__(self, prior_mu=None, prior_sigma=None):
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.post_mu = None
        self.post_sigma = None
        
    def fit(self, X, y, sigma_noise=1):
        n_features = X.shape[1]
        
        # 默认无信息先验
        if self.prior_mu is None:
            self.prior_mu = np.zeros(n_features)
        if self.prior_sigma is None:
            self.prior_sigma = 1e4 * np.eye(n_features)
            
        # 计算后验参数
        A = X.T @ X / sigma_noise**2 + np.linalg.inv(self.prior_sigma)
        self.post_sigma = np.linalg.inv(A)
        self.post_mu = self.post_sigma @ (
            X.T @ y / sigma_noise**2 + 
            np.linalg.inv(self.prior_sigma) @ self.prior_mu
        )
    
    def predict(self, X, return_std=False):
        y_mean = X @ self.post_mu
        if return_std:
            y_std = np.sqrt(np.diag(X @ self.post_sigma @ X.T))
            return y_mean, y_std
        return y_mean

# 使用示例
X_train = np.c_[np.ones(len(X)), X]  # 添加截距项
model = BayesianLinearRegression()
model.fit(X_train, y)

# 对比解析解与真实值
print(f"解析解斜率: {model.post_mu[1]:.2f} (真实值2.5)")
print(f"解析解截距: {model.post_mu[0]:.2f}")
```



### 2. 变分推断近似实现
```python
from scipy.stats import multivariate_normal

class VariationalBayesianLR:
    def __init__(self, max_iter=1000, tol=1e-5):
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 初始化变分参数
        mu = np.zeros(n_features)
        Sigma = np.eye(n_features)
        lambda_ = 1.0  # 噪声精度
        
        for _ in range(self.max_iter):
            # 更新q(beta)
            Sigma_new = np.linalg.inv(lambda_ * X.T @ X + np.eye(n_features))
            mu_new = Sigma_new @ (lambda_ * X.T @ y)
            
            # 更新q(lambda)
            a_n = 1 + n_samples/2
            b_n = 1 + 0.5 * np.sum((y - X @ mu_new)**2) 
            lambda_new = a_n / b_n
            
            # 检查收敛
            if np.linalg.norm(mu - mu_new) < self.tol:
                break
            mu, Sigma, lambda_ = mu_new, Sigma_new, lambda_new
        
        self.mu = mu
        self.Sigma = Sigma
        self.lambda_ = lambda_new
    
    def predict(self, X):
        return X @ self.mu

# 使用示例
vb_model = VariationalBayesianLR()
vb_model.fit(X_train, y)
print(f"变分推断斜率: {vb_model.mu[1]:.2f}")
```



## 四、进阶主题与最佳实践

### 1. 超参数选择
- **先验方差**：通过交叉验证选择最优 $\Sigma_0$
- **证据近似（Evidence Approximation）**：最大化边际似然
  $$
  p(y|X) = \int p(y|X,\beta)p(\beta) d\beta
  $$

### 2. 非共轭先验处理
- **哈密尔顿蒙特卡洛（HMC）**：PyMC3默认采样方法
- **平均场变分推断**：快速近似后验分布

### 3. 多输出回归扩展
$$
Y = XB + E, \quad E \sim \mathcal{N}(0, \Sigma)
$$
其中 $B$ 为系数矩阵，$\Sigma$ 为协方差矩阵



## 五、总结与扩展阅读

### 核心优势总结
- **概率解释**：量化参数和预测的不确定性
- **灵活扩展**：易与层次模型、非线性模型结合
- **自动正则化**：通过先验分布控制模型复杂度

### 推荐学习资源
- 经典教材：《Pattern Recognition and Machine Learning》第3章
- 前沿论文：Bayesian Deep Learning
- 实用工具：PyMC3官方文档
