
# 特征编码

## 1. 标签编码（Label Encoding）

### 理论深化
- 核心原理：将类别变量映射为整数标签（0到n_classes-1），保持类别顺序性。
- 数学表达：
  $$
  \text{Encode}(C_i) = i \quad \text{其中 } C_i \in \{C_1, C_2, ..., C_k\}, \ i \in \{0, 1, ..., k-1\}
  $$
- 适用场景：有序类别（如学历等级：小学、初中、高中）或树模型（如决策树可处理有序整数）。
- 局限性：引入人为顺序关系，可能误导线性模型（如逻辑回归、SVM）。

### Python实现扩展
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 示例数据
df = pd.DataFrame({'grade': ['A', 'B', 'C', 'A', 'D']})

# Scikit-learn实现
le = LabelEncoder()
df['grade_encoded'] = le.fit_transform(df['grade'])
print("标签编码结果:\n", df)

# 手动映射（保持可解释性）
label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
df['grade_manual'] = df['grade'].map(label_map)
```



## 2. 独热编码（One-Hot Encoding）

### 理论深化
- 核心原理：将类别变量转换为k维二元向量（k为类别数），仅一维为1。
- 数学表达：
  $$
  \text{Encode}(C_i) = [0, ..., 1, ..., 0] \in \{0,1\}^k
  $$
- 稀疏性优化：使用稀疏矩阵存储（尤其当k>100时）。
- 维度灾难：当k极大时（高基数特征），需配合降维技术（如PCA）。

### Python实现扩展
```python
from sklearn.preprocessing import OneHotEncoder

# Scikit-learn实现（保留DataFrame结构）
encoder = OneHotEncoder(sparse_output=False, drop='first')  # 删除第一列避免共线性
encoded = encoder.fit_transform(df[['grade']])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['grade']))
df = pd.concat([df, df_encoded], axis=1)
print("独热编码结果:\n", df.head())

# Pandas快捷实现（get_dummies）
df_dummies = pd.get_dummies(df['grade'], prefix='grade', drop_first=True)
df = pd.concat([df, df_dummies], axis=1)
```



## 3. 目标编码（Target Encoding）

### 理论深化
- 核心原理：用目标变量的统计量（均值、分位数）代替类别标签。
- 数学公式（均值编码）：
  $$
  \text{Encode}(C_i) = \frac{\sum_{j=1}^{n} y_j \cdot I(x_j = C_i)}{\sum_{j=1}^{n} I(x_j = C_i)}
  $$
- 平滑技术：防止过拟合，引入先验概率（贝叶斯平滑）：
  $$
  \text{Encode}(C_i) = \frac{\text{count}(C_i) \cdot \text{mean}(C_i) + \alpha \cdot \text{global\_mean}}{\text{count}(C_i) + \alpha}
  $$
  （$\alpha$为平滑参数，控制先验权重）
- 交叉验证：需在训练集内计算，避免数据泄露。

### Python实现扩展
```python
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split

# 生成数据
df = pd.DataFrame({
    'city': ['NY', 'LA', 'NY', 'SF', 'SF', 'LA'],
    'income': [80, 65, 72, 90, 85, 70]
})

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    df[['city']], df['income'], test_size=0.3, random_state=42
)

# 使用category_encoders库
te = TargetEncoder(smoothing=2.0)
X_train_encoded = te.fit_transform(X_train, y_train)
X_test_encoded = te.transform(X_test)

# 手动实现平滑目标编码
global_mean = y_train.mean()
alpha = 2.0

train_encoded = X_train.copy()
for col in ['city']:
    # 计算每个类别的统计量
    agg = y_train.groupby(X_train[col]).agg(['mean', 'count'])
    # 平滑公式
    train_encoded[col+'_encoded'] = (
        (agg['count'] * agg['mean'] + alpha * global_mean) 
        / (agg['count'] + alpha)
    ).reindex(X_train[col]).values

print("手动目标编码结果:\n", train_encoded)
```



## 4. 频率编码（Frequency Encoding）

### 理论深化
- 核心原理：用类别出现频率代替原始标签，反映类别分布信息。
- 数学公式：
  $$
  \text{Encode}(C_i) = \frac{\text{count}(C_i)}{n}
  $$
- 变体：对数频率（缓解长尾分布）：
  $$
  \text{Encode}(C_i) = \log\left(1 + \frac{\text{count}(C_i)}{n}\right)
  $$

### Python实现扩展
```python
# 计算频率
freq_map = df['city'].value_counts(normalize=True)
df['city_freq'] = df['city'].map(freq_map)

# 对数频率
df['city_log_freq'] = np.log(1 + df['city'].map(df['city'].value_counts()))

print("频率编码结果:\n", df[['city', 'city_freq', 'city_log_freq']])
```



## 5. 二进制编码（Binary Encoding）

### 理论深化
- 核心原理：将类别先转换为整数标签，再转化为二进制位表示。
- 步骤：
  1. Label Encoding：将类别映射为整数。
  2. 将整数转换为二进制字符串（如3 → '11'）。
  3. 按二进制位拆分列（每列为一位）。
- 优势：维度数从k降至$\log_2(k)$，适合高基数特征。

### Python实现扩展
```python
import category_encoders as ce

# 使用category_encoders库
encoder = ce.BinaryEncoder(cols=['city'])
df_binary = encoder.fit_transform(df['city'])

# 手动实现
df['city_label'] = LabelEncoder().fit_transform(df['city'])
max_bits = int(np.ceil(np.log2(df['city_label'].nunique())))
for bit in range(max_bits):
    df[f'city_bit_{bit}'] = (df['city_label'] >> bit) & 1

print("二进制编码结果:\n", df.head())
```



## 6. 效果编码（Effect Encoding）

### 理论深化
- 核心原理：类似独热编码，但将最后一类表示为-1（避免共线性）。
- 数学公式：
  $$
  \text{Encode}(C_i) = 
  \begin{cases}
  1 & \text{if } x = C_i \\
  -1 & \text{if } x = C_k \text{（最后一类）} \\
  0 & \text{otherwise}
  \end{cases}
  $$
- 优点：保留线性模型可解释性，同时减少一个维度。

### Python实现扩展
```python
# 使用statsmodels的对比编码
from patsy.contrasts import ContrastMatrix
from sklearn.preprocessing import OneHotEncoder

# 自定义效果编码矩阵
def effect_coding(n_categories):
    contrast = np.eye(n_categories - 1)
    contrast = np.vstack([contrast, -np.ones((1, n_categories - 1))])
    return contrast

encoder = OneHotEncoder(handle_unknown='ignore', drop='first')
encoded = encoder.fit_transform(df[['city']])
effect_matrix = effect_coding(3)  # 假设city有3类
df_effect = pd.DataFrame(encoded.toarray() @ effect_matrix, 
                         columns=[f'city_effect_{i}' for i in range(2)])
print("效果编码结果:\n", df_effect)
```



## 7. 哈希编码（Hashing Encoding）

### 理论深化
- 核心原理：通过哈希函数将类别映射到固定维度空间（如m维）。
- 数学公式：
  $$
  \text{Encode}(C_i) = [h_1(C_i), h_2(C_i), ..., h_m(C_i)] \in \mathbb{R}^m
  $$
  其中$h_j$为哈希函数。
- 优点：解决高基数特征内存问题，维度可控。
- 缺点：可能发生哈希冲突，损失信息。

### Python实现扩展
```python
from sklearn.feature_extraction import FeatureHasher

# 使用FeatureHasher
hasher = FeatureHasher(n_features=4, input_type='string')
hashed = hasher.transform(df['city'].apply(lambda x: [x]))
df_hash = pd.DataFrame(hashed.toarray(), columns=[f'city_hash_{i}' for i in range(4)])
print("哈希编码结果:\n", df_hash.head())
```



## 8. 分箱编码（Binning Encoding）

##### 理论深化
- 核心原理：将连续变量离散化后编码，捕捉非线性关系。
- 分箱方法：
  - 等宽分箱：按值范围均匀划分。
  - 等频分箱：按样本量均匀划分。
  - 聚类分箱：基于K-Means聚类。
  - 决策树分箱：基于信息增益或卡方检验。

### Python实现扩展
```python
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

# 生成数据
df = pd.DataFrame({'age': np.random.randint(18, 70, 100)})

# 等频分箱（5箱）
encoder = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
df['age_bin'] = encoder.fit_transform(df[['age']]).astype(int)

# 分箱后目标编码（WOE）
def calc_woe(df, feature, target):
    total_good = df[target].sum()
    total_bad = (1 - df[target]).sum()
    woe_dict = {}
    for bin in df[feature].unique():
        bin_df = df[df[feature] == bin]
        good = bin_df[target].sum()
        bad = (1 - bin_df[target]).sum()
        woe = np.log((bad / total_bad) / (good / total_good))
        woe_dict[bin] = woe
    return woe_dict

# 假设有目标变量
df['default'] = np.random.randint(0, 2, 100)
woe_map = calc_woe(df, 'age_bin', 'default')
df['age_woe'] = df['age_bin'].map(woe_map)

print("分箱WOE编码结果:\n", df[['age', 'age_bin', 'age_woe']].head())
```



## 9. 词嵌入（Word Embedding）

### 理论深化
- 核心原理：将高维稀疏类别映射到低维稠密向量（如Word2Vec、GloVe）。
- 数学形式：
  给定类别序列$C = [c_1, c_2, ..., c_n]$，学习映射：
  $$
  E: c_i \rightarrow \mathbf{v}_i \in \mathbb{R}^d
  $$
  其中d为嵌入维度。
- 训练方式：  
  - CBOW：根据上下文预测中心词。
  - Skip-Gram：根据中心词预测上下文。

### Python实现扩展
```python
from gensim.models import Word2Vec

# 示例数据（类别序列）
sentences = [
    ['NY', 'LA', 'SF'],
    ['LA', 'SF', 'NY'],
    ['SF', 'NY', 'LA']
]

# 训练Word2Vec模型
model = Word2Vec(sentences, vector_size=3, window=2, min_count=1, workers=4)

# 获取嵌入向量
ny_vector = model.wv['NY']
print("NY的嵌入向量:", ny_vector)

# 应用到DataFrame
df = pd.DataFrame({'city': ['NY', 'LA', 'SF']})
df['city_embedding'] = df['city'].apply(lambda x: model.wv[x])
```



## 10. 时间特征编码

### 理论深化
- 周期性编码：将时间分解为周期分量（如sin/cos编码）。
  $$
  \sin\left(\frac{2\pi t}{T}\right), \quad \cos\left(\frac{2\pi t}{T}\right)
  $$
  其中T为周期长度（如24小时、12个月）。
- 时间差编码：计算与参考时间的时间差（如天数）。

### Python实现扩展
```python
# 生成时间数据
df = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=24, freq='H')
})

# 周期性编码（小时）
df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)

# 时间差编码（距2023-01-01的天数）
df['days_since'] = (df['timestamp'] - pd.Timestamp('2023-01-01')).dt.days

print("时间特征编码结果:\n", df.head())
```

## 总结与选择策略

| 方法          | 适用场景                           | 优点                          | 缺点                          |
|-------------------|---------------------------------------|----------------------------------|----------------------------------|
| 标签编码          | 有序类别、树模型                      | 保留顺序，维度不变               | 引入虚假顺序关系                 |
| 独热编码          | 无序类别、类别数少                    | 无偏，可解释性强                 | 维度爆炸，内存消耗大             |
| 目标编码          | 高基数类别、有监督任务                | 捕捉类别与目标关系               | 需防过拟合，可能泄露目标信息     |
| 频率编码          | 类别分布重要、无监督任务              | 反映类别出现频率                 | 丢失类别语义信息                 |
| 二进制编码        | 高基数类别（100 < k < 1000）          | 维度适中，内存高效               | 可解释性差                       |
| 哈希编码          | 极高基数类别（k > 1000）              | 固定维度，内存稳定               | 哈希冲突导致信息损失             |
| 分箱编码          | 连续变量非线性关系                    | 捕捉非线性，抗噪                 | 信息损失，需调优分箱策略         |
| 词嵌入            | 类别间有语义关系（如地址、产品）      | 捕捉语义相似性                   | 需要足够数据训练，计算成本高     |
| 时间编码          | 时间序列特征                          | 提取周期性、趋势信息             | 需领域知识设计周期参数           |

实践建议：
1. 低基数类别（k < 10）：优先考虑独热编码或效果编码。
2. 高基数类别（k > 50）：使用目标编码、频率编码或哈希编码。
3. 文本/序列类别：尝试词嵌入或TF-IDF加权编码。
4. 时间特征：必须进行周期性编码，避免模型误解时间顺序。
5. 模型适配：
   - 树模型：标签编码、目标编码。
   - 线性模型：独热编码、效果编码。
   - 深度学习：嵌入编码、哈希编码。

```python
# 综合编码Pipeline示例
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 定义不同列的编码方式
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), ['gender']),
        ('target', TargetEncoder(), ['city']),
        ('bin', KBinsDiscretizer(n_bins=5), ['age'])
    ],
    remainder='passthrough'
)

# 构建完整Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# 使用示例
X_train = df[['gender', 'city', 'age']]
y_train = df['target']
pipeline.fit(X_train, y_train)
```