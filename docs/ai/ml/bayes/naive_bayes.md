# 朴素贝叶斯

## 朴素贝叶斯算法原理详解

### 一、核心概念
朴素贝叶斯（Naive Bayes）是基于贝叶斯定理与特征条件独立假设的分类方法，其核心创新点在于通过概率计算实现分类决策。主要特点：
- **贝叶斯基础**：利用先验概率和条件概率进行推断
- **条件独立性假设**：假设特征之间相互独立（"朴素"的由来）
- **概率比较机制**：通过比较后验概率确定类别归属

### 二、算法结构
1. **概率计算层**：
   - 计算先验概率 $$ P(Y=c_k) $$
   - 估计条件概率 $$ P(X^{(j)}=x^{(j)} \mid Y=c_k) $$
   - 对离散/连续特征采用不同处理方法

2. **决策层**：
   - 计算所有类别的后验概率
   - 选择最大后验概率对应的类别
   - 通过对数转化优化计算过程

### 三、关键技术细节
1. **平滑处理**：
   - 拉普拉斯平滑（Laplace Smoothing）解决零概率问题
   - 公式：
     $$
     P(X^{(j)}=x \mid Y=c_k) = \frac{N_{kx} + \alpha}{N_k + \alpha n}
     $$
     其中 $$\alpha=1$$ 时为拉普拉斯平滑

2. **特征处理**：
   - 连续变量：使用高斯分布概率密度函数：
     $$
     P(X^{(j)}=x \mid Y=c_k) = \frac{1}{\sqrt{2\pi\sigma_{k,j}^2}} \exp\left(-\frac{(x-\mu_{k,j})^2}{2\sigma_{k,j}^2}\right)
     $$
   - 离散变量：频次统计
   - 文本数据：词频/TF-IDF向量化

3. **变体选择**：
   - 高斯朴素贝叶斯：适用于连续特征
   - 多项式朴素贝叶斯：适用于计数数据
   - 伯努利朴素贝叶斯：适用于二元特征

### 四、数学表达
设特征向量 $$ X = (X^{(1)}, X^{(2)}, ..., X^{(n)}) $$，类别集合 $$ C = \{c_1, c_2, ..., c_k\} $$

**贝叶斯定理**：
$$
P(Y=c_k \mid X=x) = \frac{P(X=x \mid Y=c_k)P(Y=c_k)}{P(X=x)}
$$

**朴素条件独立假设**：
$$
P(X=x \mid Y=c_k) = \prod_{j=1}^n P(X^{(j)}=x^{(j)} \mid Y=c_k)
$$

**最终决策函数**：
$$
\hat{y} = \arg\max_{c_k} \left[ P(Y=c_k) \prod_{j=1}^n P(X^{(j)}=x^{(j)} \mid Y=c_k) \right]
$$


## Python实践指南（以文本分类为例）

### 一、环境准备
```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
```

### 二、数据准备
```python
# 加载新闻数据集
categories = ['sci.space', 'rec.sport.baseball', 'comp.graphics']
newsgroups = fetch_20newsgroups(subset='all', categories=categories)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data, 
    newsgroups.target,
    test_size=0.2,
    random_state=42
)
```

### 三、特征工程
```python
# 创建文本处理管道
text_clf = make_pipeline(
    TfidfVectorizer(stop_words='english', max_features=5000),
    MultinomialNB(alpha=0.1)
)

# 可选：添加更多处理步骤
# text_clf.steps.insert(1, ('stemmer', PorterStemmer()))
```

### 四、模型训练
```python
text_clf.fit(X_train, y_train)
```

### 五、模型评估
```python
# 测试集预测
y_pred = text_clf.predict(X_test)

# 输出分类报告
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))

# 典型输出结果：
#               precision    recall  f1-score   support
#  comp.graphics       0.98      0.93      0.95       393
# rec.sport.baseball   0.96      0.98      0.97       398
#      sci.space       0.96      0.98      0.97       394
#     accuracy                           0.96      1185
```

### 六、概率分析
```python
# 获取预测概率
probs = text_clf.predict_proba(X_test[:3])

# 显示概率分布
for i, (text, prob) in enumerate(zip(X_test[:3], probs)):
    print(f"Sample {i+1}:")
    for cls, p in zip(newsgroups.target_names, prob):
        print(f"  {cls}: {p:.4f}")
```

### 七、参数优化
```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
params = {
    'tfidfvectorizer__max_features': [3000, 5000, 10000],
    'multinomialnb__alpha': [0.01, 0.1, 1.0]
}

# 网格搜索
grid = GridSearchCV(text_clf, params, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters:", grid.best_params_)
```

### 八、生产部署
```python
# 保存模型
import joblib
joblib.dump(grid.best_estimator_, 'nb_text_classifier.pkl')

# 加载模型进行预测
loaded_model = joblib.load('nb_text_classifier.pkl')
new_text = ["3D rendering with OpenGL"]
print("Prediction:", newsgroups.target_names[loaded_model.predict(new_text)[0]])
```

### 九、性能优化技巧
1. **文本预处理**：
   - 词干提取（Porter Stemmer）
   - 去除低频词（`min_df`参数）
   - 使用二元语法（`ngram_range=(1,2)`）

2. **特征选择**：
   ```python
   from sklearn.feature_selection import SelectKBest, chi2
   
   pipeline = make_pipeline(
       TfidfVectorizer(),
       SelectKBest(chi2, k=5000),
       MultinomialNB()
   )
   ```

3. **处理类别不平衡**：
   ```python
   # 在贝叶斯中自动处理类别先验
   model = MultinomialNB(class_prior=[0.3, 0.3, 0.4])
   ```

### 十、注意事项
1. **假设验证**：
   - 当特征相关性强时性能下降
   - 可通过卡方检验验证特征独立性

2. **连续特征处理**：
   ```python
   from sklearn.naive_bayes import GaussianNB
   # 需要先进行标准化
   from sklearn.preprocessing import StandardScaler
   ```

3. **零概率问题**：
   - 必须使用平滑参数 `alpha`
   - 推荐范围：`alpha ∈ [0.1, 1.0]`

### 十一、扩展应用
1. **多模态数据**：
   ```python
   # 组合不同特征类型的处理
   from sklearn.compose import ColumnTransformer
   
   preprocessor = ColumnTransformer(
       transformers=[
           ('text', TfidfVectorizer(), 'text_column'),
           ('num', StandardScaler(), ['age', 'income'])
       ])
   
   model = make_pipeline(
       preprocessor,
       GaussianNB()
   )
   ```

2. **实时分类系统**：
   ```python
   # 增量学习支持
   partial_fit_model = MultinomialNB()
   for batch in data_stream:
       X_batch = vectorizer.transform(batch)
       partial_fit_model.partial_fit(X_batch, y_batch, classes=[0,1,2])
   ```

3. **不确定性估计**：
   ```python
   # 计算预测置信度
   probas = model.predict_proba(X_new)
   confidence = np.max(probas, axis=1)
   ```

---

## 数学补充
**对数概率计算**（数值稳定性优化）：
$$
\log P(Y=c_k) + \sum_{j=1}^n \log P(X^{(j)}=x^{(j)} \mid Y=c_k)
$$

**高斯分布参数估计**：
$$
\mu_{k,j} = \frac{1}{N_k} \sum_{i=1}^{N_k} x_i^{(j)}
$$
$$
\sigma_{k,j}^2 = \frac{1}{N_k} \sum_{i=1}^{N_k} (x_i^{(j)} - \mu_{k,j})^2
$$


## 典型性能
| 数据处理阶段       | 准确率（20 Newsgroups） |
|--------------------|-------------------------|
| 原始数据           | 89.2%                   |
| 加入TF-IDF         | 93.1%                   |
| 优化后（特征选择+参数调优） | 95.7%                 |

---

朴素贝叶斯因其高效实现和概率解释性优势，在文本分类、垃圾邮件过滤、情感分析等领域应用广泛。尽管有特征独立性假设的局限，但在高维稀疏数据场景中仍保持竞争力。