# 对角判别分析
## 对角判别分析（DDA）算法原理详解

### 一、核心概念
对角判别分析（Diagonal Discriminant Analysis, DDA）是线性判别分析（LDA）的简化形式，**通过假设协方差矩阵为对角矩阵来降低模型复杂度**，核心特点：
- **特征独立性假设**：各类别协方差矩阵为对角阵（特征间独立）
- **高效计算**：参数数量从 $$O(p^2)$$ 降至 $$O(p)$$
- **高维友好**：适用于特征维度 $$p$$ 远大于样本量 $$N$$ 的场景
- **鲁棒估计**：减少小样本下的参数估计方差

### 二、算法结构
1. **协方差建模**：
   - 类内协方差（共享对角矩阵）：
     $$
     \boldsymbol{\Sigma} = \text{diag}(\sigma_1^2, \sigma_2^2, ..., \sigma_p^2)
     $$
   - 对角元素估计：
     $$
     \sigma_j^2 = \frac{1}{N-C} \sum_{c=1}^C \sum_{i \in c} (x_{ij} - \mu_{cj})^2
     $$

2. **判别函数**：
   $$
   \delta_c(\mathbf{x}) = -\frac{1}{2} \sum_{j=1}^p \frac{(x_j - \mu_{cj})^2}{\sigma_j^2} + \ln P(y=c)
   $$

### 三、关键技术细节
1. **参数估计**：
   - 均值估计：
     $$
     \mu_{cj} = \frac{1}{N_c} \sum_{i \in c} x_{ij}
     $$
   - 方差收缩（防止零方差）：
     $$
     \sigma_j^2 \leftarrow \max(\sigma_j^2, \epsilon)
     $$
     （$$\epsilon=1e-6$$）

2. **分类决策**：
   - 基于马氏距离简化：
     $$
     \hat{y} = \arg\min_c \sum_{j=1}^p \frac{(x_j - \mu_{cj})^2}{\sigma_j^2} - 2\ln P(y=c)
     $$

3. **正则化扩展**：
   - 添加L2正则化：
     $$
     \sigma_j^{2, reg} = \sigma_j^2 + \lambda \cdot \text{Var}(x_j)
     $$

### 四、数学推导
**对数后验比**：
$$
\ln \frac{P(y=c|\mathbf{x})}{P(y=k|\mathbf{x})} = \sum_{j=1}^p \left[ \frac{(x_j - \mu_{kj})^2}{\sigma_j^2} - \frac{(x_j - \mu_{cj})^2}{\sigma_j^2} \right] + \ln \frac{P(y=c)}{P(y=k)}
$$

**与朴素贝叶斯关系**：
- 当假设各类独立同方差时，DDA等价于高斯朴素贝叶斯
- DDA允许各类均值不同但共享方差

---

## Python实践指南（以文本分类为例）

### 一、环境准备
```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
```

### 二、数据准备
```python
# 加载新闻数据集
categories = ['sci.med', 'comp.graphics', 'rec.sport.baseball']
newsgroups = fetch_20newsgroups(subset='all', categories=categories)

# TF-IDF向量化
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(newsgroups.data).toarray()
y = newsgroups.target

# 标准化并划分数据集
X = StandardScaler(with_mean=False).fit_transform(X)  # 保持稀疏性
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 三、DDA自定义实现
```python
class DiagonalDiscriminantAnalysis:
    def __init__(self, alpha=1e-6):
        self.alpha = alpha  # 方差平滑参数
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        
        # 计算类统计量
        self.means_ = []
        self.priors_ = []
        class_vars = []
        
        for c in self.classes_:
            X_c = X[y == c]
            self.means_.append(X_c.mean(axis=0))
            self.priors_.append(X_c.shape[0] / n_samples)
            class_vars.append(X_c.var(axis=0))
            
        # 计算共享方差（对角元素）
        self.var_ = np.sum([var * (X[y==c].shape[0]-1) 
                          for c, var in zip(self.classes_, class_vars)], axis=0)
        self.var_ /= (n_samples - len(self.classes_))
        self.var_ = np.maximum(self.var_, self.alpha)  # 方差平滑
        
        return self
    
    def predict(self, X):
        # 计算各类别得分
        scores = []
        for c in range(len(self.classes_)):
            mahalanobis = np.sum((X - self.means_[c])**2 / self.var_, axis=1)
            score = -0.5 * mahalanobis + np.log(self.priors_[c])
            scores.append(score)
        return np.argmax(np.array(scores).T, axis=1)
```

### 四、模型评估
```python
# 初始化模型
dda = DiagonalDiscriminantAnalysis(alpha=1e-6)
dda.fit(X_train, y_train)

# 预测测试集
y_pred = dda.predict(X_test)

# 计算准确率
acc = accuracy_score(y_test, y_pred)
print(f"测试集准确率：{acc:.3f}")

# 特征重要性分析
top_features = np.argsort(1 / dda.var_)[::-1][:10]
print("重要特征：", [tfidf.get_feature_names_out()[i] for i in top_features])
```

### 五、正则化调优
```python
from sklearn.model_selection import GridSearchCV

params = {'alpha': [1e-9, 1e-6, 1e-3, 1e-1]}
grid = GridSearchCV(DiagonalDiscriminantAnalysis(), 
                   params, 
                   cv=5,
                   scoring='accuracy')
grid.fit(X_train, y_train)

print(f"最优平滑参数：{grid.best_params_}")
print(f"最优准确率：{grid.best_score_:.3f}")
```

---

## 数学补充
**方差收缩推导**：
$$
\sigma_j^{2, reg} = \frac{(N-C)\sigma_j^2 + \alpha N}{N-C + \alpha}
$$

**Bhattacharyya距离简化**：
$$
D_B(c,k) = \frac{1}{4} \sum_{j=1}^p \frac{(\mu_{cj} - \mu_{kj})^2}{\sigma_j^2}
$$

---

## 注意事项
1. **假设验证**：
   - 需验证特征独立性假设合理性
   - 推荐使用卡方检验或互信息分析

2. **数据预处理**：
   - 必须进行特征标准化（如TF-IDF转换）
   - 对异常值敏感，建议使用Robust Scaling

3. **高维优化**：
   - 当 $$p > 10^4$$ 时，使用稀疏矩阵运算：
     ```python
     from scipy.sparse import csr_matrix
     X_train = csr_matrix(X_train)
     ```

---

## 扩展应用
1. **增量学习**：
   ```python
   def partial_fit(self, X_batch, y_batch):
       # 在线更新均值和方差
       for c in np.unique(y_batch):
           mask = (y_batch == c)
           X_c = X_batch[mask]
           new_mean = (self.means_[c] * self.n_c[c] + X_c.sum(0)) / (self.n_c[c] + mask.sum())
           new_var = ... # 递推方差计算
           self.means_[c] = new_mean
           self.var_ = new_var
       return self
   ```

2. **多标签扩展**：
   ```python
   from sklearn.multioutput import ClassifierChain
   dda_chain = ClassifierChain(DiagonalDiscriminantAnalysis())
   ```

3. **异构特征融合**：
   ```python
   from sklearn.pipeline import FeatureUnion
   feature_union = FeatureUnion([
       ('text', TfidfVectorizer()),
       ('meta', StandardScaler())
   ])
   ```

---

## 与相关算法对比
| 特性                | DDA               | LDA               | 朴素贝叶斯        | 逻辑回归          |
|---------------------|-------------------|-------------------|-------------------|-------------------|
| 协方差假设          | 共享对角矩阵      | 共享全矩阵        | 各类独立对角矩阵  | 无显式假设        |
| 参数数量            | $$O(p)$$         | $$O(p^2)$$       | $$O(Cp)$$        | $$O(p)$$         |
| 决策边界            | 二次曲线          | 线性              | 二次曲线          | 线性/非线性       |
| 高维性能            | 优                | 差                | 优                | 需正则化          |
| 计算复杂度          | $$O(Np)$$        | $$O(Np^2)$$      | $$O(Np)$$        | $$O(Np)$$        |

---

对角判别分析通过简化协方差结构，在文本分类、基因表达分析等高维场景中表现出色。其平衡了模型复杂度与计算效率，当特征间相关性较弱时，性能可媲美全协方差模型。与深度学习相比，DDA在小样本、高维数据中更具优势，但学习复杂非线性模式的能力有限。建议作为基线模型，为后续复杂模型提供参考。