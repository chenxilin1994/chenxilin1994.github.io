# TF-IDF

## TF-IDF 理论详解

TF-IDF（Term Frequency-Inverse Document Frequency）是一种用于信息检索和文本挖掘的经典统计方法，用于衡量一个词在文档集合中的重要性。其核心思想是：一个词在文档中出现的频率高（TF高），但在整个文档集合中出现的频率低（IDF高），则该词具有更高的区分度。


### 1. TF（Term Frequency，词频）
- 定义：词在文档中出现的频率。
- 公式：
  $$
  \text{TF}(t, d) = \frac{\text{词 } t \text{ 在文档 } d \text{ 中的出现次数}}{\text{文档 } d \text{ 的总词数}}
  $$
- 作用：衡量词对单个文档的重要性。


### 2. IDF（Inverse Document Frequency，逆文档频率）
- 定义：衡量词在整个文档集合中的稀有程度。
- 公式：
  $$
  \text{IDF}(t, D) = \log\left(\frac{\text{文档总数 } N}{\text{包含词 } t \text{ 的文档数} + 1}\right) + 1
  $$
  （加 1 是为了避免分母为零）
- 作用：降低常见词（如“的”、“是”）的权重，提升罕见词的权重。


### 3. TF-IDF 计算公式
$$
\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)
$$


## Python 实践

### 1. 使用 `sklearn` 实现 TF-IDF

```python {cmd="python3"} {cmd="python3"}
from sklearn.feature_extraction.text import TfidfVectorizer

# 示例文档集合
documents = [
    "The quick brown fox jumps over the lazy dog",
    "Never jump over the lazy dog quickly",
    "A quick brown dog outpaces a fox"
]

# 初始化 TF-IDF 向量化器
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',  # 移除英文停用词（如 "the", "a"）
    norm='l2',             # 对向量做 L2 归一化
    use_idf=True,          # 启用 IDF 计算
    smooth_idf=True        # 防止除零错误
)

# 计算 TF-IDF 矩阵
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# 获取特征词列表
feature_names = tfidf_vectorizer.get_feature_names_out()

# 输出结果
print("特征词列表:", feature_names)
print("\nTF-IDF 矩阵:\n", tfidf_matrix.toarray())
```


### 2. 代码解释
1. TfidfVectorizer 参数：
   - `stop_words`: 移除常见停用词。
   - `norm`: 归一化方式（`l2` 或 `l1`）。
   - `use_idf`: 是否使用 IDF 加权。
   - `smooth_idf`: 对 IDF 做平滑处理（避免除零错误）。

2. 输出结果：
   - `feature_names`: 所有文档中的特征词列表。
   - `tfidf_matrix`: 形状为 `(文档数, 词汇表大小)` 的矩阵，每个元素表示对应词在文档中的 TF-IDF 值。


### 3. 输出示例
```
特征词列表: ['brown' 'dog' 'fox' 'jump' 'jumps' 'lazy' 'outpaces' 'quick' 'quickly']

TF-IDF 矩阵:
 [[0.43  0.32  0.43  0.    0.55  0.32  0.    0.43  0.  ]
 [0.    0.38  0.    0.53  0.    0.38  0.    0.    0.53]
 [0.47  0.35  0.47  0.    0.    0.    0.59  0.47  0.  ]]
```


## 中文 TF-IDF 处理
中文需要先分词（英文默认按空格分词）：
```python {cmd="python3"} {cmd="python3"}
import jieba

# 中文分词示例
text = "我爱自然语言处理"
words = " ".join(jieba.cut(text))  # 输出: "我 爱 自然语言处理"

# 在 TfidfVectorizer 中指定分词器
tfidf_vectorizer = TfidfVectorizer(tokenizer=jieba.cut)
```


## 应用场景
1. 搜索引擎：计算查询词与文档的相关性。
2. 文本分类：将 TF-IDF 向量作为机器学习模型的输入。
3. 关键词提取：选择 TF-IDF 值高的词作为关键词。


## 局限性
1. 无法捕捉语义：TF-IDF 仅基于词频，忽略上下文和词序。
2. 稀疏性：高维稀疏矩阵可能影响计算效率。

通过结合深度学习（如 Word2Vec、BERT）可以弥补这些不足，但 TF-IDF 因其简单高效，仍是文本处理的基础工具。