# 岭回归

## 一、岭回归理论解析
### 1.1 背景与问题
岭回归（Ridge Regression）是一种改良的最小二乘估计法，专门用于处理以下两类问题：
1. 特征间存在高度相关性的多重共线性问题
2. 模型过拟合问题

在标准线性回归中，当设计矩阵X存在多重共线性或特征维度超过样本数时，最小二乘解会变得不稳定，甚至无法求解。此时加入L2正则化项可以改善条件数，使解更稳定。

### 1.2 核心思想
通过引入L2正则化项（惩罚项）对系数的大小进行约束：
- 限制参数θ的平方和大小
- 平衡模型复杂度与拟合能力
- 提高模型的泛化能力

## 二、数学公式推导
### 2.1 损失函数
岭回归的损失函数由最小二乘项和正则化项组成：

$J(\theta) = \frac{1}{2n}\left[ \sum_{i=1}^n (y^{(i)} - \theta^T x^{(i)})^2 + \lambda \sum_{j=1}^m \theta_j^2 \right]$

其中：
- $n$：样本数量
- $m$：特征数量
- $\lambda$：正则化强度（λ≥0）
- $\theta_j$：第j个特征系数（j≥1）

### 2.2 闭式解推导
1. 矩阵形式表示：
$J(\theta) = \frac{1}{2n}[(X\theta - y)^T(X\theta - y) + \lambda \theta^T \theta]$

2. 对θ求导并令导数为0：
$\frac{\partial J(\theta)}{\partial \theta} = \frac{1}{n}(X^T X\theta - X^T y) + \frac{\lambda}{n}\theta = 0$

3. 整理得正规方程：
$(X^T X + \lambda I)\theta = X^T y$

4. 最终解：
$\hat{\theta} = (X^T X + \lambda I)^{-1} X^T y$

其中I为(m+1)×(m+1)单位矩阵（考虑截距项时需调整）

## 三、第三方库实践（scikit-learn）
```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 加载数据
data = load_diabetes()
X, y = data.data, data.target

# 数据标准化（重要！正则化对尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 创建模型（λ=1.0）
ridge = Ridge(alpha=1.0)

# 训练模型
ridge.fit(X_train, y_train)

# 评估模型
train_score = ridge.score(X_train, y_train)
test_score = ridge.score(X_test, y_test)
print(f"训练集R²: {train_score:.3f}, 测试集R²: {test_score:.3f}")

# 查看系数
print("模型系数：", ridge.coef_)
print("截距项：", ridge.intercept_)
```

## 四、手动实现岭回归
```python
import numpy as np
from sklearn.preprocessing import StandardScaler

class ManualRidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # 正则化系数λ
        self.theta = None   # 模型参数
        self.intercept = None  # 截距项
    
    def fit(self, X, y):
        """
        训练模型
        参数:
        X : 特征矩阵，形状(n_samples, n_features)
        y : 目标向量，形状(n_samples,)
        """
        # 添加偏置项（截距项）
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # 计算正则化矩阵
        m = X_b.shape[1]
        I = np.eye(m)
        I[0, 0] = 0  # 不对截距项进行正则化
        
        # 闭式解计算
        XT_X = X_b.T.dot(X_b)
        regularization = self.alpha * I
        self.theta = np.linalg.inv(XT_X + regularization).dot(X_b.T).dot(y)
        
        # 分离截距项和系数
        self.intercept = self.theta[0]
        self.coef_ = self.theta[1:]
    
    def predict(self, X):
        """ 预测 """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)

# 使用示例
if __name__ == "__main__":
    # 使用标准化后的数据（与第三方库相同数据）
    ridge_manual = ManualRidgeRegression(alpha=1.0)
    ridge_manual.fit(X_train, y_train)
    
    # 对比结果
    print("\n手动实现结果：")
    print("系数：", ridge_manual.coef_)
    print("截距：", ridge_manual.intercept)
```

## 五、关键点解析
1. **数据标准化**：正则化对特征尺度敏感，必须对特征进行标准化处理
2. **截距项处理**：正则化项不作用于截距项（对应I矩阵的第一个元素置0）
3. **正则化强度λ**：
   - λ=0时退化为普通线性回归
   - λ→∞时所有系数趋近于0
   - 最佳λ值需要通过交叉验证确定

## 六、应用注意事项
1. **特征工程**：正则化不能替代特征选择，仍需进行必要的特征处理
2. **超参数调优**：建议使用GridSearchCV进行λ值搜索
3. **模型诊断**：结合学习曲线分析偏差-方差权衡
4. **计算效率**：当特征维度极高时，推荐使用随机梯度下降法

## 七、总结
岭回归通过引入L2正则化项，有效解决了多重共线性问题，提高了模型的泛化能力。其核心在于平衡模型复杂度和拟合能力，通过λ参数控制正则化强度。实际应用中需要特别注意数据标准化和参数调优，结合具体问题选择合适的正则化强度。

通过第三方库实现可以快速应用，而手动实现则有助于深入理解算法原理。建议在实践中结合两种方式，既保证效率又加深理论理解。