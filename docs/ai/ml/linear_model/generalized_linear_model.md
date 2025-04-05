# 广义线性模型：理论、推导与 Python 实践

广义线性模型（Generalized Linear Models, GLM）是一类统一的回归模型框架，扩展了普通线性模型的适用范围，使其能够处理非正态分布的响应变量。GLM 通过引入链接函数和指数族分布，为计数数据、二分类数据、连续正值数据等问题提供了有效的建模方法。本文将从理论基础出发，详细推导 GLM 的基本原理及参数估计方法，最后结合 Python 示例展示如何应用 GLM 进行数据建模和分析。



## 一、理论基础

### 1.1 广义线性模型的组成部分

广义线性模型由三部分组成：

1. **随机成分（Random Component）：**  
   假设响应变量$Y$服从某个指数族分布，其概率密度函数（或概率质量函数）一般形式为：
  $$
   f_Y(y;\theta,\phi) = \exp\left\{\frac{y\theta - b(\theta)}{\phi} + c(y,\phi)\right\},
  $$
   其中，$\theta$ 为自然参数，$\phi$ 为分散参数，$b(\theta)$ 和$c(y,\phi)$ 为已知函数。

2. **系统成分（Systematic Component）：**  
   线性预测器定义为：
  $$
   \eta = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p = \mathbf{x}^\top \boldsymbol{\beta},
  $$
   其中$\mathbf{x}$ 为解释变量向量，$\boldsymbol{\beta}$ 为待估计的参数向量。

3. **链接函数（Link Function）：**  
   链接函数$g(\cdot)$将响应变量的期望$\mu = E(Y)$与线性预测器$\eta$ 联系起来：
  $$
   g(\mu) = \eta.
  $$
   常见的链接函数包括逻辑函数（logit）、对数函数（log）、恒等函数等。  
   
例如，在二分类问题中，响应变量服从二项分布，常用 logit 链接函数构成逻辑回归；而在计数数据建模中，响应变量服从泊松分布，常用对数链接函数构成泊松回归。

### 1.2 最大似然估计

在 GLM 框架中，参数估计通常采用最大似然估计（MLE）方法。假设独立样本$\{(y_i, \mathbf{x}_i)\}_{i=1}^n$ 的联合似然函数为：
$$
L(\boldsymbol{\beta}, \phi) = \prod_{i=1}^{n} f_Y\big(y_i;\theta_i,\phi\big),
$$
其中$\theta_i$ 与线性预测器之间通过链接函数的反函数联系，即：
$$
\theta_i = b'^{-1}\big(g^{-1}(\eta_i)\big).
$$
取对数后得到对数似然函数：
$$
\ell(\boldsymbol{\beta},\phi) = \sum_{i=1}^{n} \left\{\frac{y_i \theta_i - b(\theta_i)}{\phi} + c(y_i,\phi)\right\}.
$$

最大似然估计的目标是找到使对数似然函数最大的参数值。



## 二、详细推导过程

### 2.1 推导 score 方程

对参数$\boldsymbol{\beta}$ 求偏导得到 score 函数。对第$j$个参数有：
$$
\frac{\partial \ell}{\partial \beta_j} = \sum_{i=1}^{n} \frac{\partial \ell_i}{\partial \theta_i} \cdot \frac{\partial \theta_i}{\partial \eta_i} \cdot \frac{\partial \eta_i}{\partial \beta_j},
$$
其中：
-$\frac{\partial \ell_i}{\partial \theta_i} = \frac{y_i - b'(\theta_i)}{\phi}$，  
-$\frac{\partial \theta_i}{\partial \eta_i} = \frac{d\theta_i}{d\mu_i}\frac{d\mu_i}{d\eta_i}$（这一步根据具体链接函数和指数族分布的形式会有特定表达），  
-$\frac{\partial \eta_i}{\partial \beta_j} = x_{ij}$。

利用$b'(\theta_i) = \mu_i$，可得到：
$$
\frac{\partial \ell}{\partial \beta_j} = \sum_{i=1}^{n} \frac{y_i - \mu_i}{\phi} \cdot \frac{d\mu_i}{d\eta_i} \, x_{ij}.
$$
其中$\frac{d\mu_i}{d\eta_i}$ 与所选的链接函数有关。例如，对于恒等链接$g(\mu)=\mu$，有$\frac{d\mu_i}{d\eta_i}=1$；而对于 logit 链接，导数则需根据$\sigma(z)(1-\sigma(z))$ 的形式计算。

设权重为$w_i = \frac{1}{\phi} \left(\frac{d\mu_i}{d\eta_i}\right)^2$，则 score 方程可写为：
$$
\sum_{i=1}^{n} (y_i - \mu_i) \left(\frac{d\mu_i}{d\eta_i}\right) x_{ij} = 0, \quad j=0,1,\dots,p.
$$
这些非线性方程组通常需要借助数值方法求解。

### 2.2 IRLS（加权最小二乘法）算法

一种常用的求解方法是迭代加权最小二乘法（Iteratively Reweighted Least Squares, IRLS），其基本思想为：
1. 初始化参数$\boldsymbol{\beta}^{(0)}$。
2. 在每次迭代中，计算当前估计下的均值$\mu_i^{(t)}$ 和导数$\frac{d\mu_i}{d\eta_i}$。
3. 定义工作响应变量$z_i$ 和权重$w_i$：
  $$
   z_i = \eta_i^{(t)} + \frac{y_i - \mu_i^{(t)}}{d\mu_i/d\eta_i}, \quad w_i = \frac{1}{\phi}\left(\frac{d\mu_i}{d\eta_i}\right)^2.
  $$
4. 更新参数，通过求解加权最小二乘问题：
  $$
   \boldsymbol{\beta}^{(t+1)} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} w_i \left(z_i - \mathbf{x}_i^\top \boldsymbol{\beta}\right)^2.
  $$
5. 重复直至收敛。

IRLS 算法将 GLM 的 MLE 问题转化为一系列加权线性回归问题，因此具有较好的数值稳定性和效率。



## 三、Python 实践

在 Python 中，可以使用 `statsmodels` 库来构建和拟合广义线性模型。下面以泊松回归（用于建模计数数据）为例，展示如何使用 GLM。

### 3.1 环境准备

首先确保安装了 `statsmodels`、`numpy` 以及 `matplotlib`：
```bash
pip install statsmodels matplotlib
```

### 3.2 示例代码

下面的代码生成模拟计数数据，并利用泊松分布的 GLM 模型进行拟合和结果分析。

```python
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 生成模拟数据
np.random.seed(42)
n = 200
x = np.linspace(0, 10, n)
# 构造线性关系，并加上泊松噪声
# 真正的模型为 log(mu) = beta0 + beta1*x, 故 mu = exp(beta0 + beta1*x)
beta0, beta1 = 1.0, 0.3
mu = np.exp(beta0 + beta1 * x)
y = np.random.poisson(mu)

# 构造数据框（利用 pandas 方便建模）
import pandas as pd
data = pd.DataFrame({'x': x, 'y': y})

# 使用 statsmodels 构建泊松回归 GLM 模型
model = smf.glm(formula='y ~ x', data=data, family=sm.families.Poisson())
result = model.fit()

print(result.summary())

# 可视化原始数据及拟合曲线
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.6, label='Observed data')
# 预测值
x_pred = np.linspace(x.min(), x.max(), 100)
df_pred = pd.DataFrame({'x': x_pred})
y_pred = result.predict(df_pred)
plt.plot(x_pred, y_pred, color='red', linewidth=2, label='Fitted GLM curve')
plt.xlabel('x')
plt.ylabel('y')
plt.title('泊松回归 - 广义线性模型拟合')
plt.legend()
plt.show()
```

### 3.3 代码说明

- **数据生成**：构造了一组服从泊松分布的计数数据，其中$\log(\mu) = \beta_0 + \beta_1 x$。
- **模型构建与拟合**：利用 `statsmodels` 中的 `glm` 接口构建泊松回归模型，并调用 `fit()` 方法进行参数估计。
- **结果输出**：使用 `summary()` 方法展示模型拟合结果，包括参数估计、标准误差、显著性检验等信息。
- **可视化**：绘制原始数据散点图，并叠加拟合的模型曲线，直观展示 GLM 的拟合效果。



## 四、总结

本文详细介绍了广义线性模型的理论基础、数学推导过程及 Python 实践：
- **理论部分**阐述了 GLM 的三大组成部分：随机成分、系统成分和链接函数，说明了其如何扩展普通线性模型以适应不同类型的响应变量。
- **推导过程**中，通过最大似然估计和 score 方程推导出参数估计的基本原理，并介绍了 IRLS 算法作为求解 GLM 的有效数值方法。
- **Python 实践**展示了如何利用 `statsmodels` 构建和拟合泊松回归模型，从而实际应用广义线性模型进行数据分析。

广义线性模型作为一个灵活而强大的建模框架，在生物统计、经济学、工程学等多个领域都有广泛应用。希望本文能帮助你全面理解 GLM 的基本原理及其在实际问题中的应用方法。