# 贝叶斯线性回归

## 贝叶斯线性回归算法原理详解

### 一、核心概念
贝叶斯线性回归（Bayesian Linear Regression）是传统线性回归的贝叶斯扩展，通过引入参数的先验分布，将参数视为随机变量进行概率推断。核心特征：
- **概率建模**：参数 $$\boldsymbol{w}$$ 服从概率分布而非固定值
- **不确定性量化**：提供预测值的置信区间
- **层次推断**：联合建模参数与超参数
- **正则化内生化**：先验分布天然起到正则化作用

### 二、算法结构
1. **先验层**：
   $$ p(\boldsymbol{w} \mid \alpha) = \mathcal{N}(\boldsymbol{w} \mid \boldsymbol{0}, \alpha^{-1}\boldsymbol{I}) $$
   - $$\alpha$$ 为精度超参数（可设置超先验）
   
2. **似然层**：
   $$ p(\boldsymbol{y} \mid \boldsymbol{X}, \boldsymbol{w}, \beta) = \mathcal{N}(\boldsymbol{y} \mid \boldsymbol{X}\boldsymbol{w}, \beta^{-1}\boldsymbol{I}) $$
   - $$\beta$$ 为噪声精度参数

3. **后验层**：
   $$ p(\boldsymbol{w} \mid \boldsymbol{X}, \boldsymbol{y}) \propto p(\boldsymbol{y} \mid \boldsymbol{X}, \boldsymbol{w}) p(\boldsymbol{w}) $$
   - 解析解为高斯分布：
     $$
     p(\boldsymbol{w} \mid \boldsymbol{X}, \boldsymbol{y}) = \mathcal{N}(\boldsymbol{w} \mid \boldsymbol{\mu}_N, \boldsymbol{\Sigma}_N)
     $$
     其中：
     $$
     \boldsymbol{\Sigma}_N^{-1} = \alpha \boldsymbol{I} + \beta \boldsymbol{X}^\top \boldsymbol{X}
     $$
     $$
     \boldsymbol{\mu}_N = \beta \boldsymbol{\Sigma}_N \boldsymbol{X}^\top \boldsymbol{y}
     $$

### 三、关键技术细节
1. **共轭先验选择**：
   - 高斯先验 × 高斯似然 ⇒ 高斯后验（解析可解）
   - 若使用非共轭先验需采用MCMC或变分推断

2. **预测分布计算**：
   $$
   p(y^* \mid \boldsymbol{x}^*, \boldsymbol{X}, \boldsymbol{y}) = \int p(y^* \mid \boldsymbol{x}^*, \boldsymbol{w}) p(\boldsymbol{w} \mid \boldsymbol{X}, \boldsymbol{y}) d\boldsymbol{w}
   $$
   解析形式：
   $$
   p(y^* \mid \boldsymbol{x}^*, \boldsymbol{X}, \boldsymbol{y}) = \mathcal{N}(y^* \mid \boldsymbol{\mu}_N^\top \boldsymbol{x}^*, \sigma_N^2(\boldsymbol{x}^*))
   $$
   其中：
   $$
   \sigma_N^2(\boldsymbol{x}^*) = \frac{1}{\beta} + (\boldsymbol{x}^*)^\top \boldsymbol{\Sigma}_N \boldsymbol{x}^*
   $$

3. **超参数优化**：
   - 证据近似（Evidence Approximation）：
     $$
     p(\alpha, \beta \mid \boldsymbol{X}, \boldsymbol{y}) \propto p(\boldsymbol{y} \mid \boldsymbol{X}, \alpha, \beta) p(\alpha) p(\beta)
     $$
   - 类型II最大似然估计：
     $$
     \gamma = \sum_{i=1}^M \frac{\lambda_i}{\lambda_i + \alpha}
     $$
     $$\lambda_i$$ 为 $$\boldsymbol{X}^\top\boldsymbol{X}$$ 的特征值

### 四、数学表达
**贝叶斯定理应用**：
$$
p(\boldsymbol{w} \mid \boldsymbol{X}, \boldsymbol{y}) = \frac{p(\boldsymbol{y} \mid \boldsymbol{X}, \boldsymbol{w}) p(\boldsymbol{w})}{p(\boldsymbol{y} \mid \boldsymbol{X})}
$$

**边际似然（模型证据）**：
$$
p(\boldsymbol{y} \mid \boldsymbol{X}) = \int p(\boldsymbol{y} \mid \boldsymbol{X}, \boldsymbol{w}) p(\boldsymbol{w}) d\boldsymbol{w}
$$

**预测方差分解**：
$$
\mathbb{V}[y^*] = \underbrace{\beta^{-1}}_{\text{观测噪声}} + \underbrace{(\boldsymbol{x}^*)^\top \boldsymbol{\Sigma}_N \boldsymbol{x}^*}_{\text{参数不确定性}}
$$

---

## Python实践指南（使用PyMC3）

### 一、环境准备
```python
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# 生成模拟数据
np.random.seed(42)
X = np.linspace(0, 1, 50)
true_w = np.array([1, -2.5, 3.0])  # 真实参数：w0 + w1*x + w2*x^2
y_true = true_w[0] + true_w[1]*X + true_w[2]*X**2
y = y_true + np.random.normal(0, 0.5, len(X))  # 添加噪声

# 构造多项式特征
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X[:, None])
```

### 二、模型定义
```python
with pm.Model() as bayesian_regression:
    # 超先验
    alpha = pm.Gamma('alpha', alpha=1, beta=1)  # 精度先验
    beta = pm.Gamma('beta', alpha=1, beta=1)    # 噪声精度
    
    # 权重先验
    w = pm.Normal('w', mu=0, sigma=1/np.sqrt(alpha), shape=X_poly.shape[1])
    
    # 线性模型
    mu = pm.math.dot(X_poly, w)
    
    # 似然
    y_obs = pm.Normal('y_obs', mu=mu, sigma=1/np.sqrt(beta), observed=y)
    
    # 采样
    trace = pm.sample(2000, tune=1000, chains=4, return_inferencedata=False)
```

### 三、后验分析
```python
# 后验分布可视化
pm.plot_posterior(trace, var_names=['w', 'beta'], 
                  credible_interval=0.95, 
                  figsize=(12, 4))
plt.show()

# 参数统计摘要
print(pm.summary(trace, var_names=['w', 'beta']))
```

### 四、预测分布
```python
# 生成新数据点
X_new = np.linspace(-0.2, 1.2, 100)
X_new_poly = poly.transform(X_new[:, None])

# 后验预测采样
with bayesian_regression:
    post_pred = pm.sample_posterior_predictive(
        trace, 
        var_names=['y_obs'],
        samples=500,
        predictions=True
    )

# 可视化预测区间
plt.figure(figsize=(10, 6))
plt.scatter(X, y, s=30, label='Observed Data')
plt.plot(X_new, true_w[0] + true_w[1]*X_new + true_w[2]*X_new**2, 
         'k--', label='True Function')

# 绘制95%置信区间
plt.plot(X_new, post_pred['y_obs'].mean(0), 'r', lw=2, label='Predictive Mean')
plt.fill_between(X_new, 
                 np.percentile(post_pred['y_obs'], 2.5, axis=0),
                 np.percentile(post_pred['y_obs'], 97.5, axis=0),
                 color='r', alpha=0.3, label='95% CI')

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
```

### 五、模型比较
```python
# 计算WAIC信息准则
waic = pm.waic(trace, bayesian_regression)
print(f"WAIC: {waic.waic:.2f} ± {waic.waic_se:.2f}")

# 比较不同阶次模型
with pm.Model() as cubic_model:
    # 定义三次多项式模型...
    # (类似步骤省略)
```

---

## 数学补充
**贝叶斯岭回归**（自动相关性确定，ARD）：
$$
p(\boldsymbol{w} \mid \boldsymbol{\alpha}) = \prod_{j=1}^M \mathcal{N}(w_j \mid 0, \alpha_j^{-1})
$$
为每个权重设置独立精度参数 $$\alpha_j$$，实现特征选择

**变分推断公式**：
$$
q(\boldsymbol{w}, \alpha, \beta) = q(\boldsymbol{w}) q(\alpha) q(\beta)
$$
通过最大化ELBO：
$$
\mathcal{L} = \mathbb{E}_q[\log p(\boldsymbol{y}, \boldsymbol{w}, \alpha, \beta)] - \mathbb{E}_q[\log q]
$$

---

## 性能优化技巧
1. **稀疏先验**：
   - 使用拉普拉斯先验实现L1正则化：
     ```python
     w = pm.Laplace('w', mu=0, b=0.1, shape=M)
     ```

2. **矩阵求逆优化**：
   - 利用Woodbury恒等式加速计算：
     $$
     (\alpha \boldsymbol{I} + \beta \boldsymbol{X}^\top \boldsymbol{X})^{-1} = \frac{1}{\alpha} \boldsymbol{I} - \frac{\beta}{\alpha^2} \boldsymbol{X}^\top (\boldsymbol{I} + \frac{\beta}{\alpha} \boldsymbol{X} \boldsymbol{X}^\top)^{-1} \boldsymbol{X}
     $$

3. **GPU加速**：
   ```python
   import aesara.tensor as at
   # 启用GPU计算
   config.floatX = 'float32'
   ```

---

## 注意事项
1. **先验敏感性**：
   - 弱信息先验（如 $$\mathcal{N}(0,10^2)$$）可能导致后验不稳定
   - 建议进行先验预测检查

2. **高维问题**：
   - 当 $$M > N$$ 时需谨慎选择先验
   - 推荐使用稀疏先验或降维技术

3. **收敛诊断**：
   ```python
   pm.plot_trace(trace)
   pm.forestplot(trace)
   ```

---

## 扩展应用
1. **贝叶斯多项式回归**：
   - 通过基函数扩展处理非线性：
     $$ \phi(\boldsymbol{x}) = [1, x, x^2, ..., x^d] $$

2. **动态线性模型**：
   ```python
   with pm.Model() as dlm:
       w = pm.GaussianRandomWalk('w', mu=0, sigma=0.1, shape=(T, M))
       y_obs = pm.Normal('y_obs', mu=at.dot(X, w.T), sigma=sigma, observed=y)
   ```

3. **异方差噪声建模**：
   ```python
   with pm.Model() as hetero_model:
       log_sigma = pm.Normal('log_sigma', mu=0, sigma=1, shape=X.shape[1])
       y_obs = pm.Normal('y_obs', mu=mu, sigma=pm.math.exp(log_sigma), observed=y)
   ```

---

## 与传统回归对比
| 特性                | 传统线性回归         | 贝叶斯线性回归           |
|---------------------|---------------------|--------------------------|
| 参数估计            | 点估计（MLE/OLS）   | 后验分布                 |
| 不确定性量化        | 仅参数协方差        | 完整预测分布             |
| 正则化              | 显式添加（L1/L2）   | 通过先验自然实现         |
| 计算复杂度          | O(M³)               | O(M³)（解析解）或更高    |
| 小样本表现          | 易过拟合            | 通过先验控制过拟合       |
| 在线学习            | 需特殊算法          | 天然支持序贯更新         |

---

贝叶斯线性回归通过概率框架提供了更丰富的建模能力，特别适合需要不确定性量化、小样本学习和模型解释性的场景。其扩展形式（如ARD、层次模型）可处理复杂现实问题，结合现代概率编程工具（PyMC3、Stan等）已成为数据科学核心方法之一。