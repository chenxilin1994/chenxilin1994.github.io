# 线性判别分析

## 线性判别分析（LDA）算法原理详解

### 一、核心概念
线性判别分析（Linear Discriminant Analysis, LDA）是一种监督学习的分类与降维方法，其核心目标是**最大化类间差异**同时**最小化类内差异**。关键特点：
- **监督投影**：利用类别标签信息寻找最优投影方向
- **判别性降维**：保留对分类最有效的低维特征
- **概率解释**：可生成类条件概率密度估计
- **多任务处理**：同时支持分类和特征提取

### 二、算法结构
1. **散布矩阵计算**：
   - 类内散布矩阵（Within-class scatter）：
     $$
     \mathbf{S}_W = \sum_{c=1}^C \sum_{i \in c} (\mathbf{x}_i - \mathbf{\mu}_c)(\mathbf{x}_i - \mathbf{\mu}_c)^T
     $$
   - 类间散布矩阵（Between-class scatter）：
     $$
     \mathbf{S}_B = \sum_{c=1}^C N_c (\mathbf{\mu}_c - \mathbf{\mu})(\mathbf{\mu}_c - \mathbf{\mu})^T
     $$
   - 总体散布矩阵：$$ \mathbf{S}_T = \mathbf{S}_W + \mathbf{S}_B $$

2. **优化目标**：
   最大化瑞利商（Rayleigh quotient）：
   $$
   J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}
   $$
   
3. **特征分解**：
   求解广义特征方程：
   $$
   \mathbf{S}_B \mathbf{w} = \lambda \mathbf{S}_W \mathbf{w}
   $$
   取前K个最大特征值对应的特征向量（K ≤ C-1）

### 三、关键技术细节
1. **多分类处理**：
   - 当类别数C > 2时，生成C-1个判别向量
   - 通过One-vs-Rest或全局散布矩阵实现

2. **正则化改进**：
   - 处理奇异矩阵问题：
     $$
     \mathbf{S}_W^{reg} = \mathbf{S}_W + \lambda \mathbf{I}
     $$
   - 调整参数λ ∈ [0,1]控制正则化强度

3. **概率建模**：
   - 类条件概率服从多元高斯分布：
     $$
     P(\mathbf{x} | y=c) = \mathcal{N}(\mathbf{x} | \mathbf{\mu}_c, \mathbf{\Sigma})
     $$
   - 后验概率计算：
     $$
     P(y=c | \mathbf{x}) \propto \exp\left( \mathbf{w}_c^T \mathbf{x} + b_c \right)
     $$

### 四、数学推导
**优化问题转换**：
通过拉格朗日乘数法将瑞利商最大化转化为：
$$
\mathbf{S}_W^{-1} \mathbf{S}_B \mathbf{w} = \lambda \mathbf{w}
$$

**投影空间**：
对于K维投影，取前K大特征值对应的特征向量组成投影矩阵：
$$
\mathbf{W} = [\mathbf{w}_1 | \mathbf{w}_2 | \cdots | \mathbf{w}_K]
$$

**分类决策函数**：
$$
\hat{y} = \arg\max_c \left[ \mathbf{w}_c^T \mathbf{x} + \ln P(y=c) \right]
$$


## Python实践指南（以葡萄酒分类为例）

### 一、环境准备
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
```

### 二、数据准备
```python
# 加载数据集
data = load_wine()
X, y = data.data, data.target

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

### 三、模型训练
```python
# 初始化LDA模型（自动选择最佳维度）
lda = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)

# 训练模型
lda.fit(X_train, y_train)

# 查看投影维度
print(f"可降维最大维度: {lda.n_components_}")
```

### 四、降维可视化
```python
# 投影到二维空间
X_proj = lda.transform(X_test)

# 绘制分类结果
plt.figure(figsize=(10,6))
scatter = plt.scatter(X_proj[:,0], X_proj[:,1], c=y_test, 
                     cmap='viridis', edgecolor='k', s=80)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.colorbar(scatter)
plt.title('LDA Projection of Wine Dataset')
plt.show()
```

### 五、分类评估
```python
# 预测测试集
y_pred = lda.predict(X_test)

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

### 六、概率分析
```python
# 获取类别概率
probs = lda.predict_proba(X_test[:5])

# 显示预测可信度
for i, (true_class, prob) in enumerate(zip(y_test[:5], probs)):
    print(f"样本{i+1}（真实类别: {data.target_names[true_class]}）")
    for cls, p in zip(data.target_names, prob):
        print(f"  {cls}: {p:.4f}")
```

### 七、参数调优
```python
# 正则化参数搜索
from sklearn.model_selection import GridSearchCV

params = {
    'solver': ['svd', 'lsqr', 'eigen'],
    'shrinkage': [None, 0.2, 0.5, 0.8, 'auto']
}

grid = GridSearchCV(LinearDiscriminantAnalysis(), 
                   param_grid=params, 
                   cv=5,
                   scoring='accuracy')
grid.fit(X_train, y_train)

print(f"最佳参数: {grid.best_params_}")
print(f"最佳准确率: {grid.best_score_:.3f}")
```


## 数学补充
**类先验概率估计**：
$$
P(y=c) = \frac{N_c}{N}
$$

**协方差矩阵计算**：
$$
\mathbf{\Sigma} = \frac{1}{N-C} \sum_{c=1}^C \sum_{i \in c} (\mathbf{x}_i - \mathbf{\mu}_c)(\mathbf{x}_i - \mathbf{\mu}_c)^T
$$

**贝叶斯决策边界**：
$$
\mathbf{w}_c^T \mathbf{x} + \ln P(y=c) = \mathbf{w}_d^T \mathbf{x} + \ln P(y=d)
$$


## 注意事项
1. **假设检验**：
   - 各类数据服从正态分布
   - 各类协方差矩阵相同（若不同则改用QDA）
   - 推荐进行Box's M检验验证协方差齐性

2. **维度限制**：
   - 最大降维数：min(C-1, p) （C为类别数，p为特征数）
   - 当特征数>样本数时需使用正则化或PCA预处理

3. **特征缩放**：
   - LDA对尺度敏感，必须进行标准化
   - 推荐使用Z-score标准化


## 扩展应用
1. **增量学习**：
   ```python
   from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
   lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
   
   # 分批次更新
   for X_batch, y_batch in data_stream:
       lda.partial_fit(X_batch, y_batch, classes=np.unique(y))
   ```

2. **核LDA（非线性扩展）**：
   ```python
   from sklearn.kernel_approximation import Nystroem
   from sklearn.pipeline import make_pipeline
   
   kernel_approx = Nystroem(kernel='rbf', n_components=100)
   lda_pipe = make_pipeline(kernel_approx, LinearDiscriminantAnalysis())
   ```

3. **多标签分类**：
   ```python
   from sklearn.multiclass import OneVsRestClassifier
   multi_lda = OneVsRestClassifier(LinearDiscriminantAnalysis())
   ```


## 与相关算法对比
| 特性                | LDA               | PCA               | QDA               |
|---------------------|-------------------|-------------------|-------------------|
| 监督性              | 是               | 否               | 是               |
| 目标函数            | 类间/类内方差比   | 方差最大化        | 类概率密度估计    |
| 适用任务            | 分类+降维        | 降维             | 分类             |
| 协方差假设          | 各类同协方差     | 无要求           | 各类异协方差     |
| 计算复杂度          | O(np² + p³)      | O(np² + p³)      | O(Cp²)           |

---

线性判别分析通过最大化类别判别信息，在保持高分类精度的同时实现有效降维。其数学基础坚实，在文本分类、生物信息学、模式识别等领域应用广泛。当数据满足高斯假设时，LDA可达到贝叶斯最优分类器性能。通过正则化改进和核方法扩展，现代LDA能够处理高维数据和复杂模式，是经典机器学习工具箱中的重要成员。