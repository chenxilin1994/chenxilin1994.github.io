# 数据划分与验证策略



### 一、数据划分与验证的核心目标
1. 评估泛化能力：确保模型在未见数据上的表现可靠。
2. 防止过拟合：避免模型过度依赖训练数据中的噪声。
3. 优化超参数：通过验证集调整模型配置。
4. 模型选择：比较不同算法或架构的性能差异。



## 二、常见数据划分与验证方法

### 1. 简单划分（Holdout Validation）
原理：将数据一次性划分为训练集、验证集和测试集。  
适用场景：数据量大、训练时间长的场景（如深度学习）。  
优点：简单高效，计算成本低。  
缺点：小数据集下结果不稳定，验证/测试集可能不具代表性。  

实现方法：
- 随机划分：直接按比例随机分割。
- 分层抽样：保持类别分布一致（分类任务）。
- 时间序列划分：按时间顺序分割（避免未来信息泄露）。

代码示例：
```python
from sklearn.model_selection import train_test_split

# 分层随机划分（分类任务）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 时间序列划分
train_size = int(len(X) * 0.6)
val_size = int(len(X) * 0.2)
X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
```



### 2. 交叉验证（Cross-Validation）
原理：将数据分为K个子集，轮流用K-1个子集训练，剩余1个验证，重复K次取平均。  
适用场景：数据量中等或较小，需稳定评估模型性能。  

#### 变体方法：
- K-Fold CV：标准K折交叉验证。
- Stratified K-Fold CV：保持每折的类别分布一致。
- Leave-One-Out (LOO)：K=样本数，每次留一个样本验证。
- Leave-P-Out：每次留P个样本验证，计算成本高。

代码示例：
```python
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut

# K-Fold（回归任务）
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

# Stratified K-Fold（分类任务）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

# Leave-One-Out
loo = LeaveOneOut()
for train_idx, val_idx in loo.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
```



### 3. 自助法（Bootstrap）
原理：有放回抽样生成训练集，未被选中的样本作为验证集（包外估计）。  
适用场景：小数据集，需充分利用样本。  
优点：适合数据量极小的场景。  
缺点：引入偏差，模型可能低估误差。  

公式：  
- 自助样本生成概率：每个样本在m次抽样中至少被选中一次的概率为 $1 - (1 - \frac{1}{n})^m \approx 1 - e^{-m/n}$。  
- 包外误差（OOB）：用未选中的样本计算模型误差。

代码示例：
```python
import numpy as np

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    oob_indices = np.setdiff1d(np.arange(n_samples), indices)
    return X[indices], y[indices], X[oob_indices], y[oob_indices]

X_train, y_train, X_val, y_val = bootstrap_sample(X, y)
```



### 4. 分层抽样（Stratified Sampling）
原理：保持划分后各子集的类别分布与原始数据一致。  
适用场景：分类任务，尤其是类别不平衡的数据集。  
实现方式：与交叉验证结合（如StratifiedKFold），或在简单划分时设置`stratify`参数。

代码示例：
```python
# 简单划分中的分层抽样
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Stratified K-Fold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
for train_idx, val_idx in skf.split(X, y):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
```



### 5. 重复交叉验证（Repeated Cross-Validation）
原理：多次运行交叉验证，减少因数据划分随机性带来的方差。  
适用场景：需要稳定评估结果的场景。  
优点：结果更可靠，减少单次划分的偶然性。  
缺点：计算成本成倍增加。

代码示例：
```python
from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
for train_idx, val_idx in rskf.split(X, y):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
```



### 6. GroupKFold
原理：确保同一组的数据不被分割到训练集和验证集。  
适用场景：数据存在分组结构（如同一患者多次测量、同一主题的多篇文章）。  

代码示例：
```python
from sklearn.model_selection import GroupKFold

groups = np.array([1, 1, 2, 2, 3, 3])  # 每个样本所属的组
gkf = GroupKFold(n_splits=3)
for train_idx, val_idx in gkf.split(X, y, groups=groups):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
```



## 三、其他补充方法

### 1. 时间序列交叉验证（TimeSeriesSplit）
原理：按时间顺序划分，训练集仅包含早于验证集的数据。  
适用场景：时间依赖数据（如股票价格、气象数据）。

代码示例：
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
```



### 2. 嵌套交叉验证（Nested Cross-Validation）
原理：外层循环评估模型性能，内层循环调参。  
适用场景：同时需要模型选择和性能评估。

代码示例：
```python
from sklearn.model_selection import GridSearchCV, cross_val_score

# 内层循环：参数调优
param_grid = {'C': [0.1, 1, 10]}
inner_cv = StratifiedKFold(n_splits=5)
clf = GridSearchCV(SVC(), param_grid, cv=inner_cv)

# 外层循环：性能评估
outer_cv = StratifiedKFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=outer_cv)
```



### 3. 分层GroupKFold
原理：结合GroupKFold和分层抽样，保持组内类别平衡。  
适用场景：分组数据且存在类别不平衡。

代码示例：
```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=3)
for train_idx, val_idx in sgkf.split(X, y, groups=groups):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
```



## 四、方法对比与选择指南

| 方法               | 适用场景                     | 优点                          | 缺点                      |
|------------------------|--------------------------------|----------------------------------|------------------------------|
| 简单划分           | 大数据、快速验证                | 计算高效                          | 小数据结果不稳定              |
| K-Fold CV          | 中等数据、通用评估              | 稳定可靠                          | 计算成本较高                  |
| Stratified K-Fold  | 分类任务、类别不平衡            | 保持类别分布                      | 仅适用于分类                  |
| GroupKFold         | 分组数据（如医学、推荐系统）    | 避免组内数据泄漏                  | 需要明确分组信息              |
| TimeSeriesSplit    | 时间序列数据                    | 符合时间依赖特性                  | 不能随机打乱数据              |
| 嵌套交叉验证       | 模型选择与评估                  | 无偏性能估计                      | 计算成本极高                  |



## 五、最佳实践总结

1. 数据量充足：优先使用简单划分（80-10-10），快速迭代模型。  
2. 中小型数据：使用分层K折交叉验证（5-10折）提高稳定性。  
3. 时间序列数据：严格按时间顺序划分，禁止打乱数据。  
4. 分组数据：使用GroupKFold或分层GroupKFold避免组内泄漏。  
5. 超参数调优：嵌套交叉验证确保参数选择无偏。  
6. 类别不平衡：始终使用分层抽样，保持数据分布一致。  
7. 计算资源有限：重复交叉验证减少到3-5次，平衡计算成本与稳定性。  



## 六、完整代码示例（综合应用）

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.metrics import accuracy_score

# 加载数据
X, y = load_iris(return_X_y=True)

# 简单划分 + 分层抽样
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 交叉验证调参
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}
clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=StratifiedKFold(5))

# 训练与验证
clf.fit(X_train, y_train)
print("最佳参数:", clf.best_params_)

# 测试集评估
y_pred = clf.predict(X_test)
print("测试集准确率:", accuracy_score(y_test, y_pred))
```
