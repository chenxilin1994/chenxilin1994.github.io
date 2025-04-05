
# 分位数回归


## 一、分位数回归理论基础

### 1. 核心概念与动机

**分位数回归（Quantile Regression）** 由Koenker和Bassett于1978年提出，突破传统均值回归的限制，可估计变量在不同分位点（如中位数、90%分位数）上的条件分布效应。适用于以下场景：
- 数据存在**异方差性**（方差非恒定）
- 关注分布尾部效应（如金融风险中的VaR计算）
- 存在显著**异常值**干扰

#### 与OLS回归的关键区别
| **特性**         | OLS回归               | 分位数回归             |
|------------------|----------------------|-----------------------|
| 优化目标         | 最小化残差平方和      | 最小化加权残差绝对值之和 |
| 估计对象         | 条件均值              | 条件分位数            |
| 对异常值敏感性   | 高                   | 低                    |
| 分布假设         | 需要正态性            | 无分布假设            |


### 2. 数学模型与损失函数

#### 分位数定义
对于随机变量 $Y$，其 $\tau$-分位数 $Q_Y(\tau)$ 满足：
$$
P(Y \leq Q_Y(\tau)) = \tau \quad (0 < \tau < 1)
$$

#### 分位数回归模型
模型表达式：
$$
Q_{Y|X}(\tau) = X\beta(\tau)
$$
- $Q_{Y|X}(\tau)$：在给定$X$下$Y$的$\tau$-条件分位数
- $\beta(\tau)$：分位数依赖的参数向量

#### 损失函数：Pinball Loss
$$
\rho_\tau(u) = u \cdot (\tau - I(u < 0)) = 
\begin{cases} 
\tau |u| & \text{if } u \geq 0 \\
(1-\tau)|u| & \text{if } u < 0 
\end{cases}
$$
目标是最小化加权绝对损失：
$$
\min_\beta \sum_{i=1}^n \rho_\tau(y_i - X_i\beta)
$$


### 3. 参数估计方法

#### 线性规划求解
将分位数回归转化为线性规划问题：
$$
\begin{align*}
\min_{\beta, u^+, u^-} & \quad \tau \sum_{i=1}^n u_i^+ + (1-\tau) \sum_{i=1}^n u_i^- \\
\text{s.t.} & \quad y_i - X_i\beta = u_i^+ - u_i^- \quad \forall i \\
& \quad u_i^+ \geq 0, \ u_i^- \geq 0 \quad \forall i
\end{align*}
$$
其中 $u_i^+ = \max(y_i - X_i\beta, 0)$，$u_i^- = \max(X_i\beta - y_i, 0)$

#### 迭代加权最小二乘法（IRLS）
通过迭代调整权重逼近解：
1. 初始化参数 $\beta^{(0)}$
2. 计算残差 $r_i = y_i - X_i\beta^{(k)}$
3. 更新权重 $w_i = \frac{\tau}{r_i}$（若 $r_i \geq 0$）或 $w_i = \frac{1-\tau}{|r_i|}$（若 $r_i < 0$）
4. 求解加权最小二乘问题：$\beta^{(k+1)} = (X^T W X)^{-1} X^T W y$
5. 重复直到收敛


## 二、Python第三方库实践

### 1. Statsmodels实现

#### 数据准备与模型训练
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 生成异方差数据
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + np.random.normal(0, 0.5 + 0.5*X, 100)  # 方差随X增大

# 训练分位数回归模型（中位数和90%分位数）
quantiles = [0.5, 0.9]
models = {}
for tau in quantiles:
    model = sm.QuantReg(y, sm.add_constant(X))
    res = model.fit(q=tau)
    models[tau] = res
```

#### 结果可视化
```python
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, label='Data')

# 绘制不同分位数回归线
x_range = np.linspace(0, 10, 100)
X_pred = sm.add_constant(x_range)

colors = {0.5: 'r', 0.9: 'g'}
for tau in quantiles:
    pred = models[tau].predict(X_pred)
    plt.plot(x_range, pred, 
             color=colors[tau], 
             linestyle='--',
             label=f'{int(tau*100)}th Quantile')

plt.xlabel('X')
plt.ylabel('y')
plt.title('分位数回归拟合效果')
plt.legend()
plt.show()
```

#### 模型结果解读
```python
# 输出中位数回归结果
print(models[0.5].summary())

# 对比不同分位数的系数
for tau in quantiles:
    print(f"\nTau={tau}:")
    print(f"截距: {models[tau].params[0]:.3f}")
    print(f"斜率: {models[tau].params[1]:.3f}")

# 示例输出：
# Tau=0.5: 截距=-0.081, 斜率=2.013
# Tau=0.9: 截距=0.542, 斜率=2.204
```


### 2. 使用LightGBM进行分位数回归
```python
import lightgbm as lgb

# 准备数据集
train_data = lgb.Dataset(X.reshape(-1,1), label=y)

# 设置分位数损失参数
params = {
    'objective': 'quantile',
    'alpha': 0.9,  # 指定分位数（0.5对应中位数）
    'metric': 'quantile',
    'learning_rate': 0.1,
    'num_leaves': 5
}

# 训练模型
gbm = lgb.train(params, train_data, num_boost_round=100)

# 预测
y_pred_gbm = gbm.predict(X.reshape(-1,1))

# 可视化对比
plt.scatter(X, y, alpha=0.3)
plt.plot(X, models[0.9].predict(sm.add_constant(X)), 'r--', label='Statsmodels')
plt.plot(X, y_pred_gbm, 'g-', label='LightGBM')
plt.legend()
plt.title('不同方法90%分位数回归对比')
plt.show()
```


## 三、手动实现分位数回归

### 1. 基于CVXPY的线性规划实现
```python
import cvxpy as cp

def quantile_regression_cvxpy(X, y, tau=0.5):
    n_samples, n_features = X.shape
    beta = cp.Variable(n_features)
    u_pos = cp.Variable(n_samples)
    u_neg = cp.Variable(n_samples)
    
    # 构建优化问题
    objective = cp.Minimize(tau * cp.sum(u_pos) + (1-tau) * cp.sum(u_neg))
    constraints = [
        y - X @ beta == u_pos - u_neg,
        u_pos >= 0,
        u_neg >= 0
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    return beta.value

# 使用示例
X_with_intercept = sm.add_constant(X).values  # 添加截距项
beta_cvx = quantile_regression_cvxpy(X_with_intercept, y, tau=0.9)
print("手动实现系数:", beta_cvx)
```

### 2. 基于Pytorch的梯度下降实现
```python
import torch
import torch.nn as nn

class QuantileRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

def quantile_loss(y_pred, y_true, tau):
    error = y_true - y_pred
    loss = torch.mean(torch.max((tau-1)*error, tau*error))
    return loss

# 训练过程
X_tensor = torch.tensor(X_with_intercept, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1,1)

model = QuantileRegressionModel(input_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(1000):
    optimizer.zero_grad()
    pred = model(X_tensor)
    loss = quantile_loss(pred, y_tensor, tau=0.9)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 获取参数
beta_torch = list(model.parameters())[0].detach().numpy()
print("PyTorch实现系数:", beta_torch)
```


## 四、高级应用与最佳实践

### 1. 分位数回归森林
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 训练分位数回归森林
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=10)
rf.fit(X.reshape(-1,1), y)

# 预测分位数
preds = {}
for quantile in [0.1, 0.5, 0.9]:
    rf.set_params(**{'min_samples_leaf': 10})  # 调整参数控制平滑度
    preds[quantile] = np.percentile(
        [tree.predict(X.reshape(-1,1)) for tree in rf.estimators_],
        quantile*100, axis=0)

# 可视化
plt.scatter(X, y, alpha=0.3)
for q in preds:
    plt.plot(X, preds[q], label=f'{int(q*100)}th')
plt.legend()
plt.title('分位数回归森林')
plt.show()
```

### 2. 模型评估与检验
#### 分位数覆盖率检验
```python
def coverage_rate(y_true, y_pred_low, y_pred_high):
    return np.mean((y_true >= y_pred_low) & (y_true <= y_pred_high))

# 计算90%预测区间覆盖率
low = models[0.05].predict(sm.add_constant(X))
high = models[0.95].predict(sm.add_constant(X))
print(f"覆盖率为: {coverage_rate(y, low, high):.2%}")
```

#### 分位数交叉验证
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
scores = []

for tau in [0.1, 0.5, 0.9]:
    fold_scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = sm.QuantReg(y_train, sm.add_constant(X_train))
        res = model.fit(q=tau)
        
        pred = res.predict(sm.add_constant(X_test))
        loss = np.mean(np.where(y_test >= pred, 
                              tau*(y_test - pred), 
                              (tau-1)*(y_test - pred)))
        fold_scores.append(loss)
    
    scores.append(np.mean(fold_scores))
    print(f"Tau={tau}: 平均损失={scores[-1]:.3f}")
```


## 五、关键问题与解决方案

### 1. 分位数交叉问题
当不同分位数的回归线交叉时，可能违反概率分布的基本性质。解决方法：
- 添加单调性约束
- 使用同时分位数回归（Simultaneous Quantile Regression）

### 2. 高维数据挑战
- **正则化分位数回归**：
  $$
  \min_\beta \sum_{i=1}^n \rho_\tau(y_i - X_i\beta) + \lambda\|\beta\|_1
  $$
  使用`statsmodels`的`QuantReg.fit_regularized()`方法

### 3. 计算效率优化
- 使用**随机坐标下降法**（SCD）
- 基于**GPU加速**的深度学习框架（如PyTorch）


## 六、行业应用案例

### 1. 金融风险管理（VaR计算）
```python
# 加载股票收益率数据
returns = pd.read_csv('stock_returns.csv', index_col=0)

# 计算5%分位数VaR
model = sm.QuantReg(returns, sm.add_constant(np.arange(len(returns))))
var_model = model.fit(q=0.05)
var = var_model.predict(sm.add_constant(np.arange(len(returns))))

plt.plot(returns, label='Daily Returns')
plt.plot(var, 'r--', label='5% VaR')
plt.fill_between(range(len(var)), var, returns.min(), color='red', alpha=0.1)
plt.legend()
plt.title('风险价值（VaR）估计')
plt.show()
```

### 2. 医疗费用预测
```python
from sklearn.datasets import fetch_openml

# 加载医疗费用数据集
medical = fetch_openml(name='medical_charges').frame

# 训练不同分位数模型
quantiles = [0.1, 0.5, 0.9]
models = {}
for tau in quantiles:
    model = sm.QuantReg(np.log(medical['charges']), 
                      sm.add_constant(medical[['age', 'bmi', 'children']]))
    res = model.fit(q=tau)
    models[tau] = res

# 可视化年龄对费用的影响
ages = np.linspace(18, 65, 100)
X_pred = sm.add_constant(pd.DataFrame({
    'age': ages,
    'bmi': 30,
    'children': 2
}))

plt.figure(figsize=(10,6))
for tau in quantiles:
    pred = np.exp(models[tau].predict(X_pred))
    plt.plot(ages, pred, label=f'{int(tau*100)}th Percentile')
plt.xlabel('Age')
plt.ylabel('Predicted Charges')
plt.title('医疗费用分位数回归分析')
plt.legend()
plt.show()
```


## 七、总结与扩展阅读

### 核心优势总结
- **全面分析分布**：揭示变量对不同分位数的影响差异
- **稳健性强**：对异常值和异方差数据更鲁棒
- **无分布假设**：不依赖误差项的正态性假设

### 推荐学习资源
- 经典教材：《Quantile Regression》Roger Koenker
- 实践指南：Statsmodels官方文档QuantReg部分
- 前沿论文：`Quantile Regression Neural Networks`
