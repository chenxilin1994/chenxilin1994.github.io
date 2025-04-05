# 弹性网络回归

弹性网络回归（Elastic Net Regression）是一种集成了岭回归（Ridge Regression）和套索回归（Lasso Regression）优点的正则化方法。它不仅能够解决多重共线性问题，还能通过 L1 正则化实现变量选择。本文将从理论基础出发，详细推导其求解过程，并结合 Python 示例展示如何进行实际应用。



## 一、理论基础

### 1.1 目标函数

假设有 $n$ 个样本和 $p$ 个特征，令 $y_i$ 为样本 $i$ 的响应值，$x_{ij}$ 为样本 $i$ 第 $j$ 个特征的取值，系数向量为 $\beta = (\beta_1, \beta_2, \ldots, \beta_p)$。弹性网络回归的目标函数可以写为：

$$
\min_{\beta} \; \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2 + \lambda \left( \alpha \sum_{j=1}^{p} |\beta_j| + \frac{1-\alpha}{2} \sum_{j=1}^{p} \beta_j^2 \right)
$$

其中：  
- $\beta_0$ 为截距项，通常可以通过数据中心化消除；  
- $\lambda > 0$ 控制正则化力度；  
- $\alpha \in [0, 1]$ 决定了 L1 和 L2 正则化的相对比例：当 $\alpha = 1$ 时退化为套索回归，$\alpha = 0$ 时退化为岭回归。

### 1.2 优缺点概述

**优点：**
- **变量选择与稀疏性：** L1 正则化部分可以将部分不重要的变量系数压缩为 0，从而实现变量选择。  
- **处理共线性：** L2 正则化部分能够减小模型参数方差，提高模型的稳定性，特别适用于特征之间存在较强相关性的情况。  
- **灵活性：** 通过调整超参数 $\lambda$ 与 $\alpha$ 能够在变量选择与模型稳定性之间取得平衡。

**局限：**
- **超参数调优复杂：** 同时需要调节 $\lambda$ 和 $\alpha$，搜索最佳组合可能较为耗时。  
- **模型解释性：** 正则化会对系数进行缩减，虽然能够实现变量选择，但系数的物理解释可能不如普通最小二乘回归直观。



## 二、详细推导过程

为了便于推导，我们假设数据已经中心化，即 $y$ 和各个 $x_j$ 均已减去均值，从而省略截距项。下面给出基于坐标下降算法求解单个系数更新的详细过程。

### 2.1 重写目标函数

固定其他变量，仅考虑单个系数 $\beta_j$ 时，定义残差：

$$
r_i^{(j)} = y_i - \sum_{k \neq j} \beta_k x_{ik}
$$

此时目标函数关于 $\beta_j$ 可写为：

$$
J(\beta_j) = \frac{1}{2n} \sum_{i=1}^{n} \left( r_i^{(j)} - \beta_j x_{ij} \right)^2 + \lambda \left( \alpha |\beta_j| + \frac{1-\alpha}{2} \beta_j^2 \right)
$$

### 2.2 求导与次梯度分析

对 $\beta_j$ 求导时需要注意，L1 正则化项 $|\beta_j|$ 在 $\beta_j = 0$ 处不可微，其导数用次梯度（subgradient）表示。当 $\beta_j \neq 0$ 时，其导数为 $\text{sign}(\beta_j)$。

计算关于 $\beta_j$ 的导数（当 $\beta_j \neq 0$）：

$$
\frac{\partial J}{\partial \beta_j} = -\frac{1}{n} \sum_{i=1}^{n} x_{ij} \left( r_i^{(j)} - \beta_j x_{ij} \right) + \lambda (1-\alpha) \beta_j + \lambda \alpha \cdot \text{sign}(\beta_j)
$$

令导数为零，即可得：

$$
-\frac{1}{n}\sum_{i=1}^{n} x_{ij} r_i^{(j)} + \frac{1}{n}\sum_{i=1}^{n} x_{ij}^2 \beta_j + \lambda (1-\alpha)\beta_j + \lambda \alpha \cdot \text{sign}(\beta_j) = 0
$$

整理关于 $\beta_j$ 的项：

$$
\left( \frac{1}{n}\sum_{i=1}^{n} x_{ij}^2 + \lambda (1-\alpha) \right)\beta_j = \frac{1}{n}\sum_{i=1}^{n} x_{ij}r_i^{(j)} - \lambda \alpha \cdot \text{sign}(\beta_j)
$$

### 2.3 引入软阈值函数

为了统一描述当 $\beta_j$ 取零或非零情况的解，我们引入软阈值函数 $S(z, \gamma)$，定义为：

$$
S(z, \gamma) = \begin{cases}
z - \gamma, & \text{若 } z > \gamma \\
z + \gamma, & \text{若 } z < -\gamma \\
0, & \text{若 } |z| \le \gamma
\end{cases}
$$

利用此函数，上述方程可以统一写为单个解：

$$
\beta_j = \frac{S\left( \frac{1}{n}\sum_{i=1}^{n} x_{ij}r_i^{(j)}, \; \lambda \alpha \right)}{\frac{1}{n}\sum_{i=1}^{n} x_{ij}^2 + \lambda (1-\alpha)}
$$

这就是当 $\beta_j \neq 0$ 时的更新公式。

### 2.4 零解条件

当 $\beta_j = 0$ 时，由于 L1 正则化项的次梯度在 $\beta_j = 0$ 取值范围为 $[-1, 1]$，要求最优性条件为：

$$
\left|\frac{1}{n}\sum_{i=1}^{n} x_{ij}r_i^{(j)}\right| \le \lambda \alpha
$$

这正说明了当上述条件满足时，软阈值函数会将 $\beta_j$ “截断”到 0，实现变量的剔除。



## 三、Python 实践

在 Python 中，利用 `scikit-learn` 库中的 `ElasticNet` 类可以非常方便地实现弹性网络回归。下面以一个示例代码说明如何生成数据、训练模型、并进行评估和可视化。

### 3.1 环境准备

首先确保已安装 `scikit-learn`（以及 `numpy`、`matplotlib` 等常用库）：

```bash
pip install scikit-learn matplotlib
```

### 3.2 示例代码

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 生成模拟数据
X, y = make_regression(n_samples=200, n_features=20, noise=10, random_state=42)

# 数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建弹性网络回归模型，alpha 控制正则化强度，l1_ratio 等同于弹性网络中的 α
model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)

# 模型评估：均方误差和 R² 得分
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("均方误差 (MSE):", mse)
print("R²得分:", r2)

# 可视化真实值与预测值对比
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('弹性网络回归：真实值与预测值对比')
plt.show()
```

### 3.3 调参技巧

在实际应用中，为获得最佳模型性能，通常需要通过交叉验证来搜索最佳的超参数组合。例如，可以使用 `GridSearchCV` 对 `alpha` 和 `l1_ratio` 进行调参：

```python
from sklearn.model_selection import GridSearchCV

# 定义超参数搜索范围
param_grid = {
    'alpha': [0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.5, 0.9]
}

# 使用 5 折交叉验证寻找最佳参数
grid_search = GridSearchCV(ElasticNet(random_state=42), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print("最佳参数组合：", grid_search.best_params_)
print("最佳交叉验证得分：", grid_search.best_score_)
```
