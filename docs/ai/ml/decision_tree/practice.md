# 决策树Python实践

## 一、环境准备与库安装
### 1. 安装依赖库
```bash
pip install scikit-learn pandas matplotlib graphviz
```

### 2. 导入必要库
```python {cmd="python3"}
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import load_iris, load_diabetes
import graphviz
import matplotlib.pyplot as plt
```


## 二、分类任务实践（以鸢尾花数据集为例）

### 1. 数据加载与预处理
```python {cmd="python3"}
# 加载数据集
iris = load_iris()
X = iris.data  # 特征矩阵 (150x4)
y = iris.target  # 标签 (0: setosa, 1: versicolor, 2: virginica)
feature_names = iris.feature_names  # 特征名
class_names = iris.target_names    # 类别名

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 2. 模型训练与预测
```python {cmd="python3"}
# 初始化决策树分类器（使用基尼指数）
clf = DecisionTreeClassifier(
    criterion='gini',      # 分裂标准（'gini'或'entropy'）
    max_depth=3,           # 最大深度（控制过拟合）
    min_samples_split=10,  # 节点继续分裂的最小样本数
    min_samples_leaf=5     # 叶节点最小样本数
)

# 训练模型
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 评估准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")  # 输出：Test Accuracy: 0.9778
```

### 3. 可视化决策树
```python {cmd="python3"}
# 导出Graphviz格式的决策树
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True
)

# 生成可视化图形
graph = graphviz.Source(dot_data)
graph.render("iris_decision_tree")  # 保存为PDF文件
graph.view()                        # 直接显示
```



## 三、回归任务实践（以糖尿病数据集为例）

### 1. 数据加载与划分
```python {cmd="python3"}
# 加载糖尿病数据集
diabetes = load_diabetes()
X = diabetes.data  # 特征矩阵 (442x10)
y = diabetes.target  # 目标值（病情进展）

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 2. 模型训练与评估
```python {cmd="python3"}
# 初始化回归树（使用均方误差）
reg = DecisionTreeRegressor(
    criterion='squared_error',  # 分裂标准
    max_depth=4,
    min_samples_split=20
)

# 训练模型
reg.fit(X_train, y_train)

# 预测
y_pred = reg.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")  # 输出：Test MSE: 2900.55
```


## 四、关键参数详解（以分类为例）
#### 1. 核心参数
| 参数名               | 作用                                                         | 示例值          |
|----------------------|------------------------------------------------------------|-----------------|
| `criterion`          | 分裂标准（分类：'gini'或'entropy'；回归：'squared_error'）  | 'gini'          |
| `max_depth`          | 树的最大深度（控制模型复杂度，防止过拟合）                   | 3               |
| `min_samples_split`  | 节点继续分裂所需的最小样本数（值越大树越简单）               | 10              |
| `min_samples_leaf`   | 叶节点所需的最小样本数（防止噪声点影响）                     | 5               |
| `max_features`       | 分裂时考虑的最大特征数（None为全部，'sqrt'为平方根）         | 'sqrt'          |
| `ccp_alpha`          | 剪枝参数（值越大树越小）                                     | 0.01            |

### 2. 剪枝策略
```python {cmd="python3"}
# 后剪枝示例（代价复杂度剪枝）
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# 通过交叉验证选择最优alpha
clf_pruned = DecisionTreeClassifier(random_state=42)
grid = GridSearchCV(
    clf_pruned,
    {'ccp_alpha': ccp_alphas},
    cv=5
)
grid.fit(X_train, y_train)

print(f"Best alpha: {grid.best_params_['ccp_alpha']:.4f}")
```


## 五、高级技巧与注意事项
### 1. 处理连续特征
- 决策树天然支持连续特征，无需离散化（CART算法自动寻找最优切分点）.

### 2. 缺失值处理
- Scikit-learn的决策树暂不支持缺失值，需提前处理：
  ```python {cmd="python3"}
  # 填充均值或中位数
  from sklearn.impute import SimpleImputer
  imputer = SimpleImputer(strategy='median')
  X_train = imputer.fit_transform(X_train)
  X_test = imputer.transform(X_test)
  ```

### 3. 类别特征编码
- 需手动将类别特征转为数值（如LabelEncoder或OneHotEncoder）：
  ```python {cmd="python3"}
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()
  y_encoded = le.fit_transform(y)
  ```

### 4. 特征重要性分析
```python {cmd="python3"}
# 获取特征重要性
importances = clf.feature_importances_

# 可视化
plt.barh(feature_names, importances)
plt.xlabel('Feature Importance')
plt.show()
```


## 六、总结
1. **核心步骤**：
   - 数据准备 → 参数配置 → 模型训练 → 评估与可视化 → 调参优化
2. **调参建议**：
   - 优先调整 `max_depth` 和 `min_samples_split` 控制模型复杂度
   - 使用网格搜索（`GridSearchCV`）自动化调参
3. **注意事项**：
   - 避免过拟合：通过预剪枝（参数限制）或后剪枝（`ccp_alpha`）
   - 类别特征需编码，缺失值需填充
   - 特征重要性分析可辅助特征工程
