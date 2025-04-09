# AdaBoost算法

AdaBoost（Adaptive Boosting）是一种高效的集成学习算法，通过组合多个弱分类器构建强分类器。其核心思想是自适应调整样本权重和分类器权重，逐步聚焦于难以正确分类的样本。以下是其原理的详细解析：

---

### 一、核心思想
1. 弱分类器（Weak Classifier）  
   如单层决策树（决策树桩），性能略高于随机猜测（错误率 < 50%）。
2. 权重调整机制  
   - 样本权重：错误分类的样本在后续迭代中获得更高权重，迫使后续分类器关注难例。
   - 分类器权重：误差率低的分类器在最终模型中占据更高权重。

---

### 二、算法流程（以二分类为例）
输入：训练集$\{(x_1, y_1), ..., (x_N, y_N)\}$，其中$y_i \in \{-1, +1\}$，弱分类器算法（如决策树桩）。

#### 步骤 1：初始化样本权重
$$
D_1(i) = \frac{1}{N}, \quad i=1,2,...,N
$$

#### 步骤 2：迭代训练 T 个弱分类器
对每轮迭代$t = 1, 2, ..., T$：
1. 训练弱分类器  
   使用当前样本权重$D_t$训练弱分类器$h_t$，目标是最小化加权错误率：
   $$
   \epsilon_t = \sum_{i=1}^N D_t(i) \cdot I\left( h_t(x_i) \neq y_i \right)
   $$
   
2. 计算分类器权重$\alpha_t$  
   $$
   \alpha_t = \frac{1}{2} \ln\left( \frac{1 - \epsilon_t}{\epsilon_t} \right)
   $$
   -$\epsilon_t$越小，$\alpha_t$越大（准确分类器权重更高）。

3. 更新样本权重  
   $$
   D_{t+1}(i) = D_t(i) \cdot \exp\left( -\alpha_t y_i h_t(x_i) \right) / Z_t
   $$
   - 正确分类：$y_i h_t(x_i) = +1$，权重降低（乘$e^{-\alpha_t}$）。
   - 错误分类：$y_i h_t(x_i) = -1$，权重升高（乘$e^{\alpha_t}$）。
   -$Z_t$为归一化因子，确保$\sum D_{t+1}(i) = 1$。

#### 步骤 3：组合强分类器
$$
H(x) = \text{sign}\left( \sum_{t=1}^T \alpha_t h_t(x) \right)
$$
- 最终分类结果为各弱分类器的加权投票。

---

### 三、关键数学推导
1. 权重更新公式的来源  
   AdaBoost 可看作对指数损失函数的前向分步优化：
   $$
   L(y, H(x)) = \exp(-y H(x))
   $$
   每轮迭代通过梯度下降最小化损失，推导出权重更新规则。

2. 分类器权重$\alpha_t$  
   通过最小化加权错误率$\epsilon_t$，推导出$\alpha_t = \frac{1}{2} \ln\left( \frac{1 - \epsilon_t}{\epsilon_t} \right)$，确保误差越小的分类器权重越高。

---

### 四、示例说明
假设训练集有 5 个样本，初始权重均为 0.2。  
- 第一轮：分类器$h_1$错误分类样本 3 和 4，错误率$\epsilon_1 = 0.4$。  
  -$\alpha_1 = \frac{1}{2} \ln\left( \frac{1 - 0.4}{0.4} \right) \approx 0.405$。  
  - 错误样本权重更新为$0.2 \cdot e^{0.405} \approx 0.3$，正确样本权重降低。  
- 后续轮次：新的分类器$h_2$将更关注样本 3 和 4。

---

### 五、优缺点与应用
优点：  
- 高精度，无需调参，不易过拟合（弱分类器简单时）。  
- 可处理各种数据类型（需适配弱分类器）。

缺点：  
- 对噪声和异常值敏感（因权重持续增加）。  
- 弱分类器太强可能导致过拟合。

应用场景：  
- 人脸检测、文本分类、特征选择等。

---
### 六、Python实践

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# 使用乳腺癌数据集（二分类问题）：
data = load_breast_cancer()
X = data.data  # 特征
y = data.target  # 标签（0: 恶性, 1: 良性）
feature_names = data.feature_names


# 划分训练集和测试集：
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 构建AdaBoost模型
# 使用决策树桩（max_depth=1）作为基分类器
# 基分类器：单层决策树（决策树桩）
base_estimator = DecisionTreeClassifier(max_depth=1)

# AdaBoost模型
model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=50,  # 迭代次数（弱分类器数量）
    learning_rate=1.0,  # 学习率（缩小每个分类器的贡献）
    random_state=42
)

# 训练模型
model.fit(X_train, y_train)

# 预测并计算指标：
y_pred = model.predict(X_test)

# 准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("混淆矩阵")
plt.show()

# 分类报告
print(classification_report(y_test, y_pred))
```

**关键代码解析**

- 1. 基分类器选择
    - `DecisionTreeClassifier(max_depth=1)` 是决策树桩，作为弱分类器。
    - 可替换为其他分类器（如逻辑回归），但需支持样本权重（通过`fit`方法的`sample_weight`参数）。

- 2. AdaBoost参数
    - `n_estimators`: 弱分类器数量（迭代次数），增加可能提升性能，但可能过拟合。
    - `learning_rate`: 学习率，控制每个分类器的权重贡献，通常与`n_estimators`联合调参（例如小学习率需更多迭代）。

- 3. 特征重要性
    AdaBoost可输出特征重要性（基于分类器权重）：
    ```python
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, model.feature_importances_)
    plt.xlabel("特征重要性")
    plt.title("AdaBoost特征重要性")
    plt.show()
    ```


**超参数调优示例**

使用网格搜索优化参数：

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [30, 50, 100],
    'learning_rate': [0.5, 1.0, 1.5]
}

grid_search = GridSearchCV(
    AdaBoostClassifier(estimator=base_estimator, random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print("最佳参数:", grid_search.best_params_)
print("最佳准确率:", grid_search.best_score_)
```
---
### 七、总结
AdaBoost 通过动态调整样本权重和分类器权重，逐步优化模型，其本质是在指数损失函数下的加法模型。理解其权重更新机制和分类器组合策略是掌握该算法的关键。