# Lasso回归

Lasso回归（Least Absolute Shrinkage and Selection Operator）是一种线性回归方法，通过引入L1正则化实现特征选择和模型正则化。其核心思想是在最小二乘损失函数的基础上增加一个系数的L1范数惩罚项，使得部分系数被压缩为零，从而简化模型。

## 1. 模型与损失函数
给定数据集 $X \in \mathbb{R}^{n \times p}$ 和响应变量 $y \in \mathbb{R}^n$，线性回归模型为：
$$
y = X\beta + \epsilon
$$
其中 $\beta \in \mathbb{R}^p$ 为系数，$\epsilon$ 为误差项。Lasso的损失函数为：
$$
L(\beta) = \frac{1}{2n} \| y - X\beta \|_2^2 + \lambda \|\beta\|_1
$$
其中，$\lambda$ 是正则化参数，控制惩罚强度。

## 2. 优化方法：坐标下降法
由于L1正则项不可导，常用坐标下降法优化。对每个系数 $\beta_j$ 依次更新，固定其他系数，求解单变量优化问题：

**步骤推导**：
1. 残差计算：  
$$
r^{(j)} = y - \sum_{k \neq j} X_{k} \beta_k
$$
2. 单变量损失函数：  
$$
L(\beta_j) = \frac{1}{2n} \| r^{(j)} - X_j \beta_j \|_2^2 + \lambda |\beta_j|
$$
3. 求导并令导数为零：  
$$
\frac{\partial L}{\partial \beta_j} = -\frac{1}{n} X_j^T (r^{(j)} - X_j \beta_j) + \lambda \cdot \text{sign}(\beta_j) = 0
$$
4. 解方程得闭式解（软阈值函数）：  
$$
\beta_j = \frac{S(X_j^T r^{(j)}, n\lambda)}{X_j^T X_j}
$$
其中，软阈值函数 $S(z, \gamma)$ 定义为：  
$$
S(z, \gamma) = \text{sign}(z) \cdot \max(|z| - \gamma, 0)
$$



## Python第三方包实现
使用`scikit-learn`的`Lasso`类：

```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np

# 生成数据
np.random.seed(0)
X = np.random.randn(100, 5)
y = X @ np.array([3, 2, 0, 0, -1]) + np.random.randn(100) * 0.5

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_centered = y - np.mean(y)

# 训练模型
alpha = 0.1  # 对应理论中的λ
lasso = Lasso(alpha=alpha, max_iter=10000, tol=1e-4)
lasso.fit(X_scaled, y_centered)

print("系数：", lasso.coef_)
```



## Python手动实现

### 代码实现

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

def soft_threshold(z, gamma):
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0)

def lasso_coordinate_descent(X, y, lambda_, max_iter=1000, tol=1e-4):
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    r = y.copy()
    X_j_norm_sq = np.sum(X**2, axis=0)
    threshold = n_samples * lambda_  # 计算软阈值参数

    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(n_features):
            X_j = X[:, j]
            beta_j_old = beta[j]
            rho_j = X_j @ r + X_j_norm_sq[j] * beta_j_old
            beta_j_new = soft_threshold(rho_j, threshold) / X_j_norm_sq[j]
            delta = beta_j_new - beta_j_old
            beta[j] = beta_j_new
            r -= X_j * delta  # 更新残差
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_centered = y - np.mean(y)

# 运行手动实现
beta_manual = lasso_coordinate_descent(X_scaled, y_centered, lambda_=0.1)
print("手动实现系数：", beta_manual)
```

### 关键点说明
1. **标准化处理**：确保特征均值为0、方差为1，避免尺度差异影响正则化效果。
2. **软阈值函数**：处理L1正则项的不可导性，实现系数的稀疏性。
3. **残差更新**：每次更新一个系数后，立即调整残差以提高效率。

