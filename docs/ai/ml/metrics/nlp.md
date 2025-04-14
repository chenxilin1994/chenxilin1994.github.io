# 自然语言处理相关评价指标详解



## 一、文本生成与机器翻译指标

### 1. BLEU（Bilingual Evaluation Understudy）
原理：基于n-gram精确率，衡量生成文本与参考文本的n-gram重叠度，引入短句惩罚（Brevity Penalty）。  
公式：
$$
\text{BLEU} = BP \cdot \exp\left( \sum_{n=1}^N w_n \log p_n \right)
$$
- $p_n$：n-gram精确率  
- $BP = \min\left(1, e^{1 - \frac{r}{c}}\right)$，其中 $r$ 为参考文本长度，$c$ 为生成文本长度  

Python实现：
```python
from nltk.translate.bleu_score import sentence_bleu
reference = [['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumps', 'over', 'the', 'sleepy', 'dog']
score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
print("BLEU Score:", score)  # 输出约0.48
```



### 2. ROUGE（Recall-Oriented Understudy for Gisting Evaluation）
原理：基于召回率，衡量生成文本与参考文本的n-gram或最长公共子序列（LCS）重叠度。  
变体：
- ROUGE-N：n-gram召回率  
  $$
  ROUGE\text{-}N = \frac{\sum_{S \in Ref} \sum_{gram_n \in S} Count_{\text{match}}(gram_n)}{\sum_{S \in Ref} \sum_{gram_n \in S} Count(gram_n)}
  $$
- ROUGE-L：最长公共子序列（LCS）的F1值  

Python实现：
```python
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score("The quick brown fox jumps over the lazy dog",
                      "The fast brown fox jumps over the sleepy dog")
print("ROUGE-1:", scores['rouge1'])  # F1值
print("ROUGE-L:", scores['rougeL'])
```



### 3. METEOR（Metric for Evaluation of Translation with Explicit ORdering）
原理：结合精确率、召回率、词干匹配、同义词匹配和句子结构惩罚。  
公式：
$$
METEOR = (1 - \text{Penalty}) \cdot \frac{10PR}{R + 9P}
$$
- $P$：精确率，$R$：召回率  
- $\text{Penalty}$：基于词序不一致性的惩罚因子  

Python实现：
```python
from nltk.translate.meteor_score import meteor_score
reference = [['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumps', 'over', 'the', 'sleepy', 'dog']
score = meteor_score(reference, candidate)
print("METEOR:", score)  # 输出约0.75
```



### 4. TER（Translation Edit Rate）
原理：计算将生成文本转换为参考文本所需的最少编辑操作次数（插入、删除、替换、调序）。  
公式：
$$
TER = \frac{\text{编辑次数}}{\text{参考文本长度}}
$$

Python实现：
```python
import pyter
ter_score = pyter.ter(candidate, reference)
print("TER:", ter_score)  # 值越小越好
```



### 5. BERTScore
原理：基于BERT的上下文嵌入，计算生成文本与参考文本的语义相似度。  
公式：
$$
\text{BERTScore} = \frac{1}{|y|} \sum_{i} \max_{j} \cos(h_{y_i}, h_{x_j})
$$
- $h_{y_i}$ 和 $h_{x_j}$ 分别为生成文本和参考文本的BERT嵌入。  

Python实现：
```python
from bert_score import score
P, R, F1 = score([candidate], [reference], lang="en")
print("BERTScore F1:", F1.mean().item())
```



## 二、文本分类与序列标注指标

### 1. 准确率（Accuracy）
公式：
$$
\text{Accuracy} = \frac{\text{正确预测数}}{\text{总样本数}}
$$

Python实现：
```python
from sklearn.metrics import accuracy_score
y_true = [0, 1, 2, 0]
y_pred = [0, 1, 1, 0]
print("Accuracy:", accuracy_score(y_true, y_pred))  # 输出 0.75
```



### 2. F1 Score（宏平均/微平均）
公式：
$$
F1 = 2 \cdot \frac{P \cdot R}{P + R}
$$

Python实现：
```python
from sklearn.metrics import f1_score
print("Macro F1:", f1_score(y_true, y_pred, average='macro'))  # 各类别平等权重
```



### 3. 混淆矩阵（Confusion Matrix）
Python实现：
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
```



## 三、命名实体识别（NER）指标

### 1. Span-based F1
原理：基于实体跨度（起始和结束位置）计算精确率、召回率和F1。  
Python实现：
```python
from seqeval.metrics import f1_score
y_true = [['B-PER', 'I-PER', 'O', 'B-ORG']]
y_pred = [['B-PER', 'O', 'O', 'B-ORG']]
print("Span F1:", f1_score(y_true, y_pred))  # 输出 0.6667
```



## 四、问答系统指标

### 1. Exact Match（EM）
原理：预测答案与真实答案完全一致的比例。  
Python实现：
```python
def exact_match(pred_answer, true_answer):
    return int(pred_answer.strip().lower() == true_answer.strip().lower())

print("EM:", exact_match("Paris", "paris"))  # 输出 1
```

### 2. F1 Score（词级）
原理：预测答案与真实答案的词级重叠F1值。  
Python实现：
```python
from sklearn.metrics import f1_score

def token_f1(pred_tokens, true_tokens):
    common = set(pred_tokens) & set(true_tokens)
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(true_tokens)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("Token F1:", token_f1(["Paris"], ["Paris"]))  # 输出 1.0
```



## 五、语言模型评估指标

### 1. 困惑度（Perplexity, PPL）
原理：衡量模型对测试数据的概率分布的预测能力，值越小越好。  
公式：
$$
PPL = \exp\left( -\frac{1}{N} \sum_{i=1}^N \log p(w_i | w_{<i}) \right)
$$

Python实现（基于Hugging Face）：
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(inputs, labels=inputs["input_ids"])
ppl = torch.exp(outputs.loss).item()
print("Perplexity:", ppl)  # 输出约30-50
```



## 六、文本相似度与语义相关指标

### 1. 余弦相似度（Cosine Similarity）
原理：计算两段文本嵌入向量的余弦夹角相似度。  
Python实现：
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vec1 = np.array([0.8, 0.2, 0.5]).reshape(1, -1)
vec2 = np.array([0.7, 0.3, 0.6]).reshape(1, -1)
print("Cosine Similarity:", cosine_similarity(vec1, vec2)[0][0])  # 输出约0.98
```

### 2. 词移距离（Word Mover’s Distance, WMD）
原理：基于词嵌入，计算两段文本间的最小语义转移成本。  
Python实现：
```python
from gensim.similarities import WmdSimilarity
model = ...  # 加载预训练词向量
wmd_sim = WmdSimilarity(texts, model.wv)
print("WMD:", wmd_sim[query])
```



## 七、对话系统评估指标

### 1. 多样性（Distinct-n）
原理：统计生成文本中不同n-gram的比例，值越高多样性越好。  
Python实现：
```python
def distinct_n(tokens, n=2):
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return len(set(ngrams)) / len(tokens) if len(tokens) >=n else 0

tokens = ["how", "are", "you", "how", "are", "you"]
print("Distinct-2:", distinct_n(tokens, n=2))  # 输出 0.333
```

### 2. 连贯性（Coherence Score）
原理：基于主题模型或上下文相关性评估对话逻辑一致性（需自定义或使用预训练模型）。  



## 八、指标对比与选择建议

| 指标                | 适用任务               | 优点                          | 缺点                      |
|---------------------|-----------------------|-----------------------------|--------------------------|
| BLEU            | 机器翻译/文本生成      | 计算简单，广泛使用            | 忽略语义，对词序不敏感      |
| ROUGE-L         | 摘要生成              | 捕捉句子结构相似性            | 依赖参考文本质量            |
| BERTScore       | 语义相似度评估         | 基于上下文语义，更贴近人类评估  | 计算资源消耗大              |
| Perplexity      | 语言模型评估           | 直接反映模型概率质量           | 无法评估生成多样性          |
| Exact Match     | 问答系统              | 严格匹配答案正确性            | 对同义表达不敏感            |



## 九、总结
- 生成任务：联合使用BLEU、ROUGE、METEOR和BERTScore，综合评估生成质量。  
- 分类任务：优先选择宏平均F1和混淆矩阵，处理类别不平衡问题。  
- 问答系统：结合EM和词级F1，平衡严格匹配和模糊匹配需求。  
- 语义相似度：使用余弦相似度或WMD，结合业务场景选择嵌入模型。  
- 实际应用：自动指标需配合人工评估（如流畅性、相关性评分），确保全面性。