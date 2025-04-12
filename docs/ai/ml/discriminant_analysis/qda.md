# 二次判别分析

## 二次判别分析（QDA）算法原理详解

### 一、核心概念
二次判别分析（Quadratic Discriminant Analysis, QDA）是线性判别分析（LDA）的非线性扩展，**允许不同类别具有独立协方差结构**，核心特点：
- **异方差建模**：各类别独立协方差矩阵 ⇒ 二次决策边界
- **概率分类器**：基于贝叶斯定理生成后验概率
- **灵活拟合**：可捕捉复杂类别分布模式
- **参数敏感性**：需更多样本估计协方差参数

### 二、算法结构
1. **类条件概率建模**：
   - 假设各类数据服从多元高斯分布：
     $$P(\mathbf{x} | y=c) = \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)$$
   - 协方差矩阵$\boldsymbol{\Sigma}_c$类别相关

2. **后验概率计算**：
   $$
   P(y=c | \mathbf{x}) = \frac{P(\mathbf{x} | y=c)P(y=c)}{\sum_{k=1}^C P(\mathbf{x} | y=k)P(y=k)}
   $$

3. **决策函数**：
   $$
   \delta_c(\mathbf{x}) = -\frac{1}{2} \ln|\boldsymbol{\Sigma}_c| - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_c)^T \boldsymbol{\Sigma}_c^{-1} (\mathbf{x}-\boldsymbol{\mu}_c) + \ln P(y=c)
   $$
   预测类别：$\hat{y} = \arg\max_c \delta_c(\mathbf{x})$

### 三、关键技术细节
1. **协方差估计**：
   - 无偏估计：
     $$
     \boldsymbol{\Sigma}_c = \frac{1}{N_c-1} \sum_{i \in c} (\mathbf{x}_i - \boldsymbol{\mu}_c)(\mathbf{x}_i - \boldsymbol{\mu}_c)^T
     $$
   - 收缩正则化（Shrinkage）：
     $$
     \boldsymbol{\Sigma}_c^{reg} = (1-\alpha)\boldsymbol{\Sigma}_c + \alpha \text{tr}(\boldsymbol{\Sigma}_c)\mathbf{I}/p
     $$

2. **决策边界特性**：
   - 二次曲面：椭圆、双曲面或抛物面
   - 复杂度由协方差矩阵差异决定

3. **计算优化**：
   - 对数行列式优化：
     $$
     \ln|\boldsymbol{\Sigma}_c| = \sum_{i=1}^p \ln \lambda_i^{(c)}
     $$
     （$\lambda_i^{(c)}$为$\boldsymbol{\Sigma}_c$的特征值）
   - 矩阵求逆稳定性：使用伪逆或Cholesky分解

### 四、数学推导
**判别函数展开**：
$$
\delta_c(\mathbf{x}) = -\frac{1}{2}\mathbf{x}^T\boldsymbol{\Sigma}_c^{-1}\mathbf{x} + \mathbf{x}^T\boldsymbol{\Sigma}_c^{-1}\boldsymbol{\mu}_c - \frac{1}{2}\boldsymbol{\mu}_c^T\boldsymbol{\Sigma}_c^{-1}\boldsymbol{\mu}_c - \frac{1}{2}\ln|\boldsymbol{\Sigma}_c| + \ln P(y=c)
$$

**与LDA对比**：
- LDA假设 $\boldsymbol{\Sigma}_c = \boldsymbol{\Sigma}$ ⇒ 线性项合并
- QDA保留二次项 ⇒ 决策边界非线性


## Python实践指南（以鸢尾花分类为例）

### 一、环境准备
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
```

### 二、数据准备
```python
# 加载数据集
data = load_iris()
X, y = data.data, data.target

# 标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

### 三、模型训练
```python
# 初始化QDA模型（启用正则化）
qda = QuadraticDiscriminantAnalysis(reg_param=0.1, store_covariance=True)

# 训练模型
qda.fit(X_train, y_train)

# 查看各类协方差矩阵
for i, cov in enumerate(qda.covariance_):
    print(f"类别{i}协方差矩阵条件数：{np.linalg.cond(cov):.1e}")
```

### 四、分类评估
```python
# 预测测试集
y_pred = qda.predict(X_test)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=data.target_names,
           yticklabels=data.target_names)
plt.title('Confusion Matrix')
plt.show()

# 分类报告
print(classification_report(y_test, y_pred, target_names=data.target_names))
```

### 五、决策边界可视化
```python
# 选择两个特征可视化
X_2d = X_scaled[:, [0, 2]]
qda_2d = QuadraticDiscriminantAnalysis().fit(X_2d, y)

# 生成网格点
x_min, x_max = X_2d[:, 0].min()-1, X_2d[:, 0].max()+1
y_min, y_max = X_2d[:, 1].min()-1, X_2d[:, 1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# 预测网格点
Z = qda_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 绘制结果
plt.figure(figsize=(10,6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
scatter = plt.scatter(X_2d[:,0], X_2d[:,1], c=y, 
                     edgecolor='k', s=50, cmap='viridis')
plt.xlabel('标准化特征1')
plt.ylabel('标准化特征3')
plt.title('QDA决策边界')
plt.colorbar(scatter)
plt.show()
```

### 六、正则化调优
```python
from sklearn.model_selection import GridSearchCV

params = {
    'reg_param': [0, 0.1, 0.3, 0.5, 0.7, 0.9],  # 正则化强度
    'tol': [1e-4, 1e-3, 1e-2]  # 协方差估计阈值
}

grid = GridSearchCV(QuadraticDiscriminantAnalysis(),
                   param_grid=params,
                   cv=5,
                   scoring='accuracy')
grid.fit(X_train, y_train)

print(f"最优参数：{grid.best_params_}")
print(f"测试集准确率：{grid.score(X_test, y_test):.3f}")
```


## 数学补充
**马氏距离**：
$$
D_c(\mathbf{x}) = (\mathbf{x}-\boldsymbol{\mu}_c)^T \boldsymbol{\Sigma}_c^{-1} (\mathbf{x}-\boldsymbol{\mu}_c)
$$

**Bhattacharyya距离**（类别可分性度量）：
$$
D_{B}(c,k) = \frac{1}{8}(\boldsymbol{\mu}_c - \boldsymbol{\mu}_k)^T \left( \frac{\boldsymbol{\Sigma}_c + \boldsymbol{\Sigma}_k}{2} \right)^{-1} (\boldsymbol{\mu}_c - \boldsymbol{\mu}_k) + \frac{1}{2} \ln \left( \frac{|\frac{\boldsymbol{\Sigma}_c + \boldsymbol{\Sigma}_k}{2}|}{\sqrt{|\boldsymbol{\Sigma}_c||\boldsymbol{\Sigma}_k|}} \right)
$$


## 注意事项
1. **数据假设验证**：
   - 各类数据需近似服从多元正态分布
   - 推荐进行Q-Q图检验或Mardia检验
   
2. **小样本问题**：
   - 当 $$N_c < p$$ 时协方差矩阵奇异
   - 解决方案：正则化、特征选择、PCA降维

3. **计算复杂度**：
   - 参数数量：$$ O(Cp^2) $$
   - 高维数据需谨慎使用（如p>100）


## 扩展应用
1. **增量学习**：
   ```python
   # 分批次更新参数
   for X_batch, y_batch in data_stream:
       qda.partial_fit(X_batch, y_batch, classes=np.unique(y))
   ```

2. **概率校准**：
   ```python
   from sklearn.calibration import CalibratedClassifierCV
   calibrated_qda = CalibratedClassifierCV(QuadraticDiscriminantAnalysis(), 
                                         method='isotonic', 
                                         cv=3)
   ```

3. **多标签扩展**：
   ```python
   from sklearn.multioutput import ClassifierChain
   qda_chain = ClassifierChain(QuadraticDiscriminantAnalysis())
   ```


## 与相关算法对比
| 特性                | QDA               | LDA               | 朴素贝叶斯        |
|---------------------|-------------------|-------------------|-------------------|
| 协方差假设          | 各类异协方差      | 各类同协方差      | 特征独立          |
| 决策边界            | 二次曲面          | 超平面            | 线性/非线性       |
| 参数复杂度          | $$O(Cp^2)$$      | $$O(p^2)$$       | $$O(Cp)$$        |
| 过拟合风险          | 高                | 低                | 中                |
| 适用场景            | 复杂类别分布      | 线性可分数据      | 高维稀疏数据      |

---

二次判别分析通过灵活建模类别协方差差异，能够捕捉复杂的非线性分类边界，在生物特征识别、金融风险分析等领域表现优异。其性能高度依赖于数据是否满足高斯假设，当特征维度较高时需配合正则化技术。与支持向量机、随机森林等现代算法相比，QDA具有概率输出的优势，但计算复杂度随维度平方增长，需权衡模型复杂度与收益。