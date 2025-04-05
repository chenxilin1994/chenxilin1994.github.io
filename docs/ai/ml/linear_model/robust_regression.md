
# 稳健回归


## 一、稳健回归理论基础

### 1. 核心思想与适用场景
**稳健回归（Robust Regression）** 是针对传统最小二乘法对异常值敏感问题发展出的改进方法，通过**修改损失函数**或**数据选择策略**提高模型鲁棒性。主要应用场景：
- 数据包含显著异常值（Outliers）
- 误差项服从重尾分布（如柯西分布）
- 存在测量误差或数据污染

#### 与传统OLS对比
| **特性**          | OLS                  | 稳健回归             |
|-------------------|----------------------|----------------------|
| 目标函数          | 平方损失             | 鲁棒损失函数         |
| 异常值敏感性      | 极高                 | 低                   |
| 计算复杂度        | O(n^3)               | 通常更高             |
| 参数估计方法      | 解析解               | 迭代优化             |



### 2. 主流方法数学原理

#### (1) M估计（M-Estimation）
通过修改损失函数降低异常值影响：
$$
\min_\beta \sum_{i=1}^n \rho(r_i) \quad \text{其中} \ r_i = y_i - x_i^T\beta
$$
常见鲁棒损失函数：
- **Huber损失**：  
  $$
  \rho(r) = 
  \begin{cases} 
  \frac{1}{2}r^2 & |r| \leq \delta \\
  \delta(|r| - \frac{1}{2}\delta) & |r| > \delta 
  \end{cases}
  $$
- **Tukey双权重**：  
  $$
  \rho(r) = 
  \begin{cases} 
  \frac{c^2}{6}\left[1 - \left(1 - (\frac{r}{c})^2\right)^3\right] & |r| \leq c \\
  \frac{c^2}{6} & |r| > c 
  \end{cases}
  $$

#### (2) RANSAC（随机抽样一致）
迭代过程：
1. 随机选取最小样本集拟合模型
2. 根据阈值识别内点（Inliers）
3. 用所有内点重新估计模型
4. 选择内点最多模型作为最终解

#### (3) Theil-Sen估计
基于中位数的非参数方法：
$$
\hat{\beta}_j = \text{median} \left( \frac{y_i - y_k}{x_{ij} - x_{kj}} \right) \quad \forall i < k
$$
时间复杂度：O(n²)，适用于小数据集



## 二、Python第三方库实践

### 1. Scikit-Learn实现

#### (1) Huber回归
```python
from sklearn.linear_model import HuberRegressor

# 生成含异常值数据
np.random.seed(42)
X = np.random.rand(100, 1)
y = 3*X.ravel() + np.random.randn(100)
y[::10] += 20  # 添加异常值

# 训练模型
huber = HuberRegressor(epsilon=1.35)  # 默认epsilon=1.35
huber.fit(X, y)

# 对比OLS
from sklearn.linear_model import LinearRegression
ols = LinearRegression().fit(X, y)

# 可视化
X_test = np.linspace(0, 1, 100).reshape(-1,1)
plt.scatter(X, y, label='Data')
plt.plot(X_test, huber.predict(X_test), 'r', label='Huber')
plt.plot(X_test, ols.predict(X_test), 'g--', label='OLS')
plt.legend()
plt.title('Huber回归 vs OLS')
plt.show()
```

#### (2) RANSAC回归
```python
from sklearn.linear_model import RANSACRegressor

ransac = RANSACRegressor(
    min_samples=0.5,  # 最小样本比例
    residual_threshold=5.0,  # 内点阈值
    stop_probability=0.99
)
ransac.fit(X, y)

# 获取内点掩码
inlier_mask = ransac.inlier_mask_

plt.scatter(X[inlier_mask], y[inlier_mask], c='b', label='Inliers')
plt.scatter(X[~inlier_mask], y[~inlier_mask], c='r', label='Outliers')
plt.plot(X_test, ransac.predict(X_test), 'k', label='RANSAC')
plt.legend()
plt.title('RANSAC回归')
plt.show()
```

#### (3) Theil-Sen回归
```python
from sklearn.linear_model import TheilSenRegressor

theil = TheilSenRegressor(n_subsamples=50, random_state=42)
theil.fit(X, y)

plt.scatter(X, y)
plt.plot(X_test, theil.predict(X_test), 'r', label='Theil-Sen')
plt.title('Theil-Sen回归')
plt.show()
```



### 2. Statsmodels实现（M估计）
```python
import statsmodels.api as sm

# 构建模型
robust_model = sm.RLM(y, sm.add_constant(X), 
                     M=sm.robust.norms.HuberT())
result = robust_model.fit()

# 结果输出
print(result.summary())
# 输出包含稳健标准误和Huber权重

# 可视化权重
plt.stem(result.weights)
plt.title('Huber权重分布（低权重对应异常值）')
plt.show()
```



## 三、手动实现关键算法

### 1. Huber回归梯度下降实现
```python
class HuberRegressorManual:
    def __init__(self, delta=1.35, max_iter=1000, lr=0.01):
        self.delta = delta
        self.max_iter = max_iter
        self.lr = lr
        self.coef_ = None
        
    def _huber_loss(self, r):
        return np.where(np.abs(r) <= self.delta, 
                       0.5*r**2, 
                       self.delta*(np.abs(r) - 0.5*self.delta))
    
    def fit(self, X, y):
        X_b = np.c_[np.ones(X.shape[0]), X]  # 添加截距项
        n_samples, n_features = X_b.shape
        self.coef_ = np.zeros(n_features)
        
        for _ in range(self.max_iter):
            residuals = y - X_b.dot(self.coef_)
            # Huber梯度
            mask = np.abs(residuals) <= self.delta
            gradient = -2 * X_b.T.dot(mask * residuals) / n_samples
            gradient -= (2 * self.delta * X_b.T.dot((~mask) * np.sign(residuals))) / n_samples
            
            self.coef_ -= self.lr * gradient
            
    def predict(self, X):
        X_b = np.c_[np.ones(X.shape[0]), X]
        return X_b.dot(self.coef_)

# 使用示例
huber_manual = HuberRegressorManual(delta=1.35)
huber_manual.fit(X, y)
print("手动实现系数:", huber_manual.coef_)
```



### 2. RANSAC核心逻辑实现
```python
def ransac_manual(X, y, n=10, k=100, t=2.0, d=50):
    best_model = None
    best_inliers = None
    max_inliers = 0
    
    for _ in range(k):
        # 随机选择样本
        sample_idx = np.random.choice(len(X), n, replace=False)
        X_sample = X[sample_idx]
        y_sample = y[sample_idx]
        
        # 拟合临时模型
        model = LinearRegression().fit(X_sample, y_sample)
        
        # 计算残差
        y_pred = model.predict(X)
        residuals = np.abs(y - y_pred)
        
        # 识别内点
        inliers = residuals < t
        n_inliers = np.sum(inliers)
        
        if n_inliers > max_inliers:
            max_inliers = n_inliers
            best_inliers = inliers
    
    # 用最佳内点重新训练
    final_model = LinearRegression().fit(X[best_inliers], y[best_inliers])
    return final_model, best_inliers

# 使用示例
ransac_model, inliers = ransac_manual(X, y)
```



## 四、方法对比与选择指南

### 1. 性能对比矩阵
| **方法**      | 异常值比例容忍度 | 计算效率 | 适用维度 | 是否需要调参 |
|---------------|------------------|----------|----------|--------------|
| Huber回归     | 中等（<30%）     | 高       | 高维     | 需选delta    |
| RANSAC        | 高（<50%）       | 低       | 低维     | 需设阈值     |
| Theil-Sen     | 中等（<30%）     | 极低     | 低维     | 无           |
| M估计（Tukey）| 中等（<30%）     | 中       | 高维     | 需选c参数    |

### 2. 参数选择建议
- **Huber的delta**：通常取1.35（覆盖95%正态数据）
- **RANSAC阈值t**：根据残差分布设定，如1.5倍标准差
- **Tukey的c**：常用4.685（效率95%）



## 五、工业级最佳实践

### 1. 异常值检测预处理
```python
from sklearn.ensemble import IsolationForest

# 使用孤立森林检测异常值
iso = IsolationForest(contamination=0.1)
outlier_mask = iso.fit_predict(X) == -1

# 可视化异常值
plt.scatter(X[~outlier_mask], y[~outlier_mask], c='b', label='正常点')
plt.scatter(X[outlier_mask], y[outlier_mask], c='r', label='异常值')
plt.legend()
plt.show()
```

### 2. 组合策略
```python
# 先RANSAC去除非结构化异常值，再用Huber回归
ransac = RANSACRegressor(residual_threshold=5.0)
ransac.fit(X, y)

X_clean = X[ransac.inlier_mask_]
y_clean = y[ransac.inlier_mask_]

huber = HuberRegressor().fit(X_clean, y_clean)
```



## 六、评估与验证

### 1. 鲁棒性评估指标
- **崩溃点（Breakdown Point）**：模型失效所需异常值最小比例
- **相对效率**：与OLS在无异常值时的效率比
- **中位数绝对误差（MAD）**：  
  $$
  \text{MAD} = \text{median}(|y_i - \hat{y}_i|)
  $$

### 2. 交叉验证改进
```python
from sklearn.model_selection import cross_val_score

# 使用鲁棒损失函数进行交叉验证
def mad_scorer(estimator, X, y):
    pred = estimator.predict(X)
    return -np.median(np.abs(y - pred))  # 负号因为sklearn最大化分数

scores = cross_val_score(HuberRegressor(), X, y, 
                        scoring=mad_scorer, cv=5)
print("MAD分数:", -scores.mean())
```



## 七、扩展应用与前沿方向

### 1. 高维稳健回归
- **稀疏稳健回归**：结合L1正则化
  ```python
  from sklearn.linear_model import RANSACRegressor
  from sklearn.linear_model import Lasso
  
  ransac_lasso = RANSACRegressor(
      base_estimator=Lasso(alpha=0.1),
      min_samples=0.5
  )
  ```

### 2. 深度稳健回归
```python
import torch
import torch.nn as nn

class RobustNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.fc(x)
    
# 使用Huber损失训练
criterion = nn.HuberLoss(delta=1.35)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
```



## 八、总结与资源推荐

### 1. 关键结论
- **Huber回归**：平衡效率与鲁棒性的首选
- **RANSAC**：适合低维数据且异常值结构化分布
- **Theil-Sen**：小数据集理论保障但计算昂贵

### 2. 推荐资源
- 经典教材：《Robust Statistical Methods》R. Maronna
- 论文："Robust Regression via Hard Thresholding"
- 工具库：Scikit-Learn Robustness文档
