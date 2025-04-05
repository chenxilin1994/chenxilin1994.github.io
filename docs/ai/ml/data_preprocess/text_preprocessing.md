
# 文本数据预处理技术深度解析


## 1. 基础清洗与标准化

### 1.1 特殊字符处理
- 数学形式化：定义清洗函数 $f: \Sigma^* \rightarrow \Sigma'^*$，其中：
  $$
  \Sigma' = \Sigma \setminus \{c_1, c_2, ..., c_k\} \quad (c_i \in \text{非文字字符})
  $$
- 正则表达式实现：
  ```python
  import re
  
  def clean_text(text):
      # 移除非字母数字字符（保留重音字母）
      cleaned = re.sub(r"[^\w\sÀ-ÿ]", "", text, flags=re.UNICODE)
      # 合并连续空白
      cleaned = re.sub(r"\s+", " ", cleaned)
      return cleaned.strip()
  ```

### 1.2 文本规范化
- Unicode规范化：
  $$
  \text{NFC}(s) = \text{Canonical Decomposition} \circ \text{Canonical Composition}(s)
  $$
- 大小写处理：
  $$
  f_{\text{case}}(w) = \begin{cases}
  \text{lower}(w) & \text{通用场景} \\
  \text{保留原样} & \text{命名实体识别}
  \end{cases}
  $$
  ```python
  import unicodedata
  text = unicodedata.normalize('NFC', text).lower()
  ```



## 2. 分词技术

### 2.1 英文分词
- 最大匹配算法：
  $$
  \arg \max_{k} \exists w \in \mathcal{D}, \text{s.t.} \quad w = t_{i:i+k}
  $$
  其中 $\mathcal{D}$ 为词典

### 2.2 中文分词
- 条件随机场（CRF）建模：
  $$
  P(y|x) = \frac{1}{Z(x)} \exp\left( \sum_{i} \theta_i f_i(y_{i-1}, y_i, x) \right)
  $$
  其中特征函数 $f_i$ 包含：
  - 字符类型（汉字/数字/符号）
  - n-gram上下文
  - 词典匹配

#### Python实现（Jieba分词）
```python
import jieba
import jieba.posseg as pseg

# 精确模式
seg_list = jieba.cut("自然语言处理是人工智能的重要方向")
print("/ ".join(seg_list))  # 自然语言/ 处理/ 是/ 人工智能/ 的/ 重要/ 方向

# 词性标注
words = pseg.cut("我爱北京天安门")
for word, flag in words:
    print(f"{word} ({flag})")
```

### 2.3 子词分词（Subword Tokenization）

#### BPE（Byte-Pair Encoding）算法
1. 初始化：将文本分解为字符级词汇表
2. 迭代合并：
   $$
   (x, y) = \arg \max_{(x,y)} \text{count}(xy) \quad \text{在所有相邻符号对中}
   $$
3. 停止条件：达到预设词汇量大小

#### WordPiece数学形式
合并准则：
$$
\text{score}(A, B) = \frac{\text{count}(AB)}{\text{count}(A) \times \text{count}(B)}
$$

#### Python实现
```python
from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=["corpus.txt"], vocab_size=30000)
encoded = tokenizer.encode("自然语言处理")
print(encoded.tokens)  # ['自', '然', '语言', '处理']
```



## 3. 词形归一化

### 3.1 词干提取（Stemming）
- Porter算法步骤：
  1. 处理复数形式：sses → ss
  2. 去除-ing结尾：walking → walk
  3. 转换副词后缀：ization → ize

- 数学规则示例：
  $$
  \text{stem}(w) = \begin{cases}
  w[:-3] & \text{if } w \text{ ends with 'ing' ∧ |w| >5} \\
  w[:-2] & \text{if } w \text{ ends with 'ed' ∧ |w| >4} \\
  w & \text{otherwise}
  \end{cases}
  $$

### 3.2 词形还原（Lemmatization）
- 基于词典的映射：
  $$
  \text{lemma}(w) = \arg \min_{l \in \mathcal{L}} \text{edit\_distance}(w, l)
  $$
  其中 $\mathcal{L}$ 为词典词根集合

#### Python实现
```python
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

# 需要指定词性
print(lemmatizer.lemmatize("running", pos=wordnet.VERB))  # run
print(lemmatizer.lemmatize("geese"))  # goose
```



## 4. 停用词处理

### 4.1 基于信息论的停用词判定
- TF-IDF权重：
  $$
  w_{t,d} = \text{tf}(t,d) \times \log \frac{N}{\text{df}(t)}
  $$
  当$w_{t,d} < \theta$时视为停用词

### 4.2 自定义停用词集
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

corpus = ["this is a sample document", "another document example"]

# 计算TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_

# 自动选择低IDF词作为停用词
stop_ids = np.where(idf < 0.5)[0]
stop_words = [vectorizer.get_feature_names_out()[i] for i in stop_ids]
```



## 5. 向量化表示

### 5.1 词袋模型（BoW）
- 文档-词项矩阵：
  $$
  M_{ij} = \text{count}(w_j \in d_i)
  $$
- TF-IDF变形：
  $$
  M_{ij}^{\text{tfidf}} = \text{tf}_{ij} \times \log \frac{N}{\text{df}_j}
  $$

### 5.2 词嵌入（Word Embedding）

#### Word2Vec的Skip-Gram目标
最大化概率：
$$
\prod_{w \in \text{Corpus}} \prod_{c \in \text{Context}(w)} P(c|w;\theta)
$$
其中：
$$
P(c|w;\theta) = \frac{\exp(v_c \cdot v_w)}{\sum_{c' \in V} \exp(v_{c'} \cdot v_w)}
$$

#### GloVe的损失函数
$$
J = \sum_{i,j=1}^V f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2
$$
其中 $f(x) = \begin{cases} (x/x_{\max})^\alpha & x < x_{\max} \\ 1 & \text{otherwise} \end{cases}$

### 5.3 上下文感知嵌入

#### BERT的MLM目标
对输入序列$X = (x_1, ..., x_T) $，随机mask 15%的token，优化：
$$
\mathcal{L} = \sum_{i \in \text{masked}} \log P(x_i | X_{\setminus i})
$$

#### 位置编码公式
$$
PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})
$$
$$
PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})
$$



## 6. 高级预处理技术

### 6.1 依存句法分析
- Arc-Eager解析算法：
  $$
  \text{Transition System} = (\Sigma, \mathcal{T}, \mathcal{C})
  $$
  其中 $\Sigma$ 为栈-缓冲区状态，$\mathcal{T}$ 包含SHIFT, LEFT-ARC, RIGHT-ARC等操作

### 6.2 语义角色标注
- BiLSTM-CRF模型：
  $$
  P(y|x) = \frac{\exp(\sum_{t=1}^T (W_{y_t} h_t + b_{y_{t-1},y_t}))}{\sum_{y'} \exp(\sum_{t=1}^T (W_{y'_t} h_t + b_{y'_{t-1},y'_t}))}
  $$
  其中 $h_t = \text{BiLSTM}(x)_t$



## 完整预处理Pipeline实现

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    
    def clean(self, text):
        text = re.sub(r"[^a-zA-ZÀ-ÿ]", " ", text)
        return text.lower()
    
    def tokenize(self, text):
        return word_tokenize(text)
    
    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(tok) for tok in tokens]
    
    def process(self, text):
        cleaned = self.clean(text)
        tokens = self.tokenize(cleaned)
        lemmas = self.lemmatize(tokens)
        return " ".join(lemmas)

# 构建处理流水线
pipeline = Pipeline([
    ('preprocess', TfidfVectorizer(
        tokenizer=TextPreprocessor().process,
        stop_words='english',
        max_features=5000
    )),
    # 可添加分类器等后续步骤
])

# 示例应用
corpus = ["Natural Language Processing is amazing!", "Text preprocessing is crucial."]
X_transformed = pipeline.fit_transform(corpus)
print(X_transformed.shape)  # (2, 5000)
```



## 技术选型矩阵

| 技术          | 适用场景                  | 计算复杂度      | 可解释性  |
|-------------------|-----------------------------|-------------------|-------------|
| 词袋模型           | 短文本分类                    | $O(NV)$        | 高          |
| TF-IDF            | 信息检索                      | $O(NV \log V)$ | 中          |
| Word2Vec          | 语义相似度计算                | $O(NVd)$       | 低          |
| BERT              | 深层语义理解任务              | $O(NT^2d)$     | 极低        |
| 依存句法分析       | 关系抽取/问答系统             | $O(NT^3)$      | 高          |



## 评估与优化

1. 词汇表截断策略：
   $$
   V_{\text{optimal}} = \arg \min_{V} \left( \text{Perplexity}(V) + \lambda |V| \right)
   $$

2. 嵌入维度选择：
   通过奇异值分解确定最优维度：
   $$
   \text{保留能量} = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^d \sigma_i^2} \geq 0.95
   $$

3. 预处理效果评估：
   - 重构误差：$\|X_{\text{orig}} - X_{\text{processed}}\|_F$
   - 下游任务指标：准确率/F1值的提升
