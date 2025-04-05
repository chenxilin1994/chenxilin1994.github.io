# 多项式回归全面解析：从数学原理到Python实现



## 一、理论基础

### 1. 定义
**多项式回归（Polynomial Regression）** 是线性回归的扩展形式，通过引入特征的**高阶项（如平方项、立方项）**来捕捉数据中的非线性关系。尽管模型对特征是非线性的，但关于参数仍然是线性的，因此仍可用线性回归方法求解。

#### 模型
给定原始特征 $x$，构造多项式特征后模型为：
$$
y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_d x^d + \epsilon
$$
- $d$：多项式次数（Degree）
- $\beta_0, \beta_1, \dots, \beta_d$：模型参数
- $\epsilon$：误差项

#### 与线性回归的关系
- **线性回归**：$y = \beta_0 + \beta_1 x$
- **多项式回归**：将 $x$ 替换为多项式基函数 $ [x, x^2, \dots, x^d]$，形式上转化为多元线性回归



### 2. 数学推导与本质

#### 特征空间映射
将原始特征 $x$ 映射到高维空间：
$$
\phi(x) = [1, x, x^2, \dots, x^d]^T \in \mathbb{R}^{d+1}
$$
模型转化为：
$$
y = \beta^T \phi(x) + \epsilon \quad \text{其中} \ \beta = [\beta_0, \beta_1, \dots, \beta_d]^T
$$

#### 参数求解
使用最小二乘法，构造设计矩阵 $X$：
$$
X = \begin{bmatrix}
1 & x_1 & x_1^2 & \dots & x_1^d \\
1 & x_2 & x_2^2 & \dots & x_2^d \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_n & x_n^2 & \dots & x_n^d
\end{bmatrix}
$$
参数解仍为：
$$
\beta = (X^T X)^{-1} X^T y
$$



### 3. 核心问题与挑战

#### 过拟合与欠拟合
- **欠拟合**（Underfitting）：多项式次数 $d$ 太小，无法捕捉数据规律  
  （例：用直线拟合抛物线数据）
- **过拟合**（Overfitting）：$d$ 太大，模型过度适应训练噪声  
  （例：用高次多项式完美拟合训练数据但泛化差）

#### 模型复杂度与泛化
- **偏差-方差权衡**：  
  $$
  \text{泛化误差} = \text{偏差}^2 + \text{方差} + \text{噪声}
  $$
  - 低次多项式：高偏差，低方差  
  - 高次多项式：低偏差，高方差



## 二、Python第三方库实践

### 1. Scikit-Learn实现流程

#### 生成模拟数据
```python
import numpy as np
import matplotlib.pyplot as plt

# 生成非线性数据（抛物线）
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X**2 + X + 2 + np.random.normal(0, 1, (100, 1))

plt.scatter(X, y, alpha=0.6)
plt.title("原始数据（抛物线分布）")
plt.show()
```

#### 多项式特征生成
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# 构建多项式回归流水线
poly_degree = 2  # 尝试修改为1（线性）、3、10观察效果
model = Pipeline([
    ('poly', PolynomialFeatures(degree=poly_degree)),
    ('linear', LinearRegression())
])

# 训练模型
model.fit(X, y)

# 预测并评估
X_test = np.linspace(-3, 3, 300).reshape(-1, 1)
y_pred = model.predict(X_test)

print(f"系数：{model.named_steps['linear'].coef_}")
print(f"截距：{model.named_steps['linear'].intercept_}")
```

#### 可视化不同次数的拟合效果
```python
degrees = [1, 2, 5, 10]
plt.figure(figsize=(12, 8))

for i, degree in enumerate(degrees):
    ax = plt.subplot(2, 2, i+1)
    
    # 训练模型
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X, y)
    y_pred = model.predict(X_test)
    
    # 绘图
    ax.scatter(X, y, alpha=0.3, label='训练数据')
    ax.plot(X_test, y_pred, 'r', label='模型预测')
    ax.set_title(f"Degree {degree}\nMSE: {mean_squared_error(y, model.predict(X)):.2f}")
    ax.legend()

plt.tight_layout()
plt.show()
```



### 2. 关键问题处理

#### 过拟合解决方案
1. **交叉验证选择最佳次数**：
```python
from sklearn.model_selection import cross_val_score

degrees = range(1, 11)
cv_scores = []

for d in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=d)),
        ('linear', LinearRegression())
    ])
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-np.mean(scores))

# 可视化选择最佳次数
plt.plot(degrees, cv_scores, marker='o')
plt.xlabel("多项式次数")
plt.ylabel("交叉验证MSE")
plt.title("模型复杂度与泛化性能")
plt.show()
```

2. **正则化（岭回归）**：
```python
from sklearn.linear_model import Ridge

model = Pipeline([
    ('poly', PolynomialFeatures(degree=10)),
    ('ridge', Ridge(alpha=100))  # 正则化强度
])
model.fit(X, y)
```
## 三、手动实现多项式回归

### 1. 基于正规方程的实现
```python
class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.coef_ = None
        self.intercept_ = None
        
    def _create_poly_features(self, X):
        """生成多项式特征矩阵"""
        n_samples = X.shape[0]
        X_poly = np.ones((n_samples, 1))  # 截距项列
        
        for d in range(1, self.degree + 1):
            X_poly = np.hstack((X_poly, X**d))
            
        return X_poly
    
    def fit(self, X, y):
        X_poly = self._create_poly_features(X)
        XT_X = X_poly.T.dot(X_poly)
        XT_y = X_poly.T.dot(y)
        
        theta = np.linalg.inv(XT_X).dot(XT_y)
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        
    def predict(self, X):
        X_poly = self._create_poly_features(X)
        return X_poly.dot(np.r_[self.intercept_, self.coef_])
```

#### 使用示例
```python
# 训练手动实现模型
pr = PolynomialRegression(degree=2)
pr.fit(X, y)

# 预测与可视化
X_test = np.linspace(-3, 3, 300).reshape(-1, 1)
y_pred = pr.predict(X_test)

plt.scatter(X, y, alpha=0.3)
plt.plot(X_test, y_pred, 'r', lw=2)
plt.title("手动多项式回归 (degree=2)")
plt.show()
```

### 2. 基于梯度下降的优化
```python
class PolynomialGradientDescent:
    def __init__(self, degree=2, learning_rate=0.01, n_iters=1000):
        self.degree = degree
        self.lr = learning_rate
        self.n_iters = n_iters
        self.theta = None  # 包含截距项和系数的完整参数
        
    def _create_poly_features(self, X):
        """生成多项式特征矩阵（包含截距项）"""
        X_poly = np.ones((X.shape[0], 1))
        for d in range(1, self.degree + 1):
            X_poly = np.hstack((X_poly, X**d))
        return X_poly
    
    def fit(self, X, y):
        X_poly = self._create_poly_features(X)
        n_samples, n_features = X_poly.shape
        
        # 参数初始化
        self.theta = np.random.randn(n_features)
        
        # 梯度下降迭代
        for _ in range(self.n_iters):
            y_pred = X_poly.dot(self.theta)
            gradient = (1/n_samples) * X_poly.T.dot(y_pred - y)
            self.theta -= self.lr * gradient
            
    def predict(self, X):
        X_poly = self._create_poly_features(X)
        return X_poly.dot(self.theta)
```



## 四、工业级最佳实践

### 1. 特征工程优化
- **数据标准化**：多项式特征可能具有极大尺度差异  
  ```python
  from sklearn.preprocessing import StandardScaler
  
  pipeline = Pipeline([
      ('poly', PolynomialFeatures(degree=3)),
      ('scaler', StandardScaler()),  # 必须放在线性模型前
      ('model', LinearRegression())
  ])
  ```

- **交互特征**：引入不同特征的乘积项  
  ```python
  PolynomialFeatures(degree=2, interaction_only=True)  # 只生成交互项
  ```

### 2. 模型解释方法
- **SHAP值分析**：解释各阶项对预测的贡献  
  ```python
  import shap
  
  # 创建解释器
  explainer = shap.LinearExplainer(model.named_steps['linear'], 
                                 model.named_steps['poly'].transform(X))
  shap_values = explainer.shap_values(X)
  
  # 可视化
  shap.summary_plot(shap_values, X, feature_names=['x', 'x²', 'x³'])
  ```



## 五、总结与扩展

### 1. 核心要点
- **适用场景**：数据呈现明显非线性趋势时  
- **关键参数**：多项式次数 $d$（需通过交叉验证选择）  
- **风险控制**：必须配合正则化防止过拟合

### 2. 进阶方向
- **样条回归（Spline Regression）**：分段多项式拟合  
- **局部加权回归（LOESS）**：基于邻近点的多项式拟合  
- **广义加性模型（GAM）**：自动学习非线性关系

### 3. 数学补充证明
**为什么多项式回归仍属于线性模型？**  
- 线性性指参数 $\beta$ 的线性，而非特征 $x$ 的线性  
- 模型可表示为 $y = \beta^T \phi(x)$，其中 $\phi(x)$ 是非线性基函数
