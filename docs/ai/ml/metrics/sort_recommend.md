# 排序与推荐任务评价指标

## 一、排序任务核心指标

### 1. 平均精度均值（Mean Average Precision, MAP）
**原理**：对每个查询的排序结果计算平均精度（AP），再对所有查询的AP取平均。  
**公式**：
- **Precision@k**：前k个结果中相关文档的比例：
  $$
  P@k = \frac{\text{前k个结果中的相关文档数}}{k}
  $$
- **Average Precision (AP)**：
  $$
  AP = \frac{1}{R} \sum_{k=1}^N P@k \cdot rel_k
  $$
  - $R$：总相关文档数  
  - $rel_k$：位置k的文档是否相关（1相关，0不相关）  
- **MAP**：
  $$
  MAP = \frac{1}{Q} \sum_{q=1}^Q AP(q)
  $$

**Python实现**：
```python
from sklearn.metrics import average_precision_score

# 示例：单个查询的相关性标签（1相关，0不相关）
y_true = [1, 0, 1, 0, 1]
y_score = [0.9, 0.8, 0.7, 0.6, 0.5]
ap = average_precision_score(y_true, y_score)
print("Average Precision:", ap)
```



### 2. 归一化折损累计增益（NDCG, Normalized Discounted Cumulative Gain）
**原理**：衡量排序结果的相关性质量，考虑位置折扣和理想排序的归一化。  
**公式**：
- **累计增益（CG）**：
  $$
  CG@k = \sum_{i=1}^k rel_i
  $$
- **折损累计增益（DCG）**：
  $$
  DCG@k = \sum_{i=1}^k \frac{rel_i}{\log_2(i+1)}
  $$
- **理想DCG（IDCG）**：按真实相关性降序排列的DCG。
- **NDCG@k**：
  $$
  NDCG@k = \frac{DCG@k}{IDCG@k}
  $$

**Python实现**：
```python
import numpy as np

def ndcg_score(y_true, y_score, k=5):
    # 按预测得分排序的索引
    order = np.argsort(y_score)[::-1]
    # 按真实相关性排序的索引（理想排序）
    ideal_order = np.argsort(y_true)[::-1]
    
    # 计算DCG@k
    dcg = sum(y_true[order[:k]] / np.log2(np.arange(2, k+2)))
    # 计算IDCG@k
    idcg = sum(y_true[ideal_order[:k]] / np.log2(np.arange(2, k+2)))
    return dcg / idcg if idcg > 0 else 0

# 示例：相关性得分（如评分或点击行为）
y_true = np.array([3, 2, 3, 0, 1])  # 真实相关性
y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])  # 模型预测得分
print("NDCG@5:", ndcg_score(y_true, y_score, k=5))
```



### 3. 平均倒数排名（Mean Reciprocal Rank, MRR）
**原理**：仅考虑第一个相关文档的位置，取倒数后求平均。  
**公式**：
$$
MRR = \frac{1}{Q} \sum_{q=1}^Q \frac{1}{\text{rank}_q}
$$
- $\text{rank}_q$：第q个查询中第一个相关文档的位置。

**Python实现**：
```python
def mrr_score(y_true_list, y_pred_ranks):
    """
    :param y_true_list: 每个查询的相关文档位置列表，如[[0, 2], [1, 3]]
    :param y_pred_ranks: 预测的排序列表，如[[2,0,1], [3,1,2]]
    """
    reciprocal_ranks = []
    for true_pos, pred_rank in zip(y_true_list, y_pred_ranks):
        for pos, doc in enumerate(pred_rank, 1):
            if doc in true_pos:
                reciprocal_ranks.append(1 / pos)
                break
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0

# 示例
y_true_list = [[0, 2], [1, 3]]  # 相关文档ID
y_pred_ranks = [[2, 0, 1], [3, 1, 2]]  # 预测的排序
print("MRR:", mrr_score(y_true_list, y_pred_ranks))  # 输出 (1/2 + 1/1)/2 = 0.75
```



## 二、推荐任务核心指标

### 1. 精确率@K（Precision@K）
**原理**：推荐列表中前K个物品中用户实际感兴趣的占比。  
**公式**：
$$
Precision@K = \frac{\text{推荐的前K个物品中相关数}}{K}
$$

**Python实现**：
```python
def precision_at_k(y_true, y_pred, k):
    # y_true: 用户真实感兴趣的物品集合
    # y_pred: 推荐列表
    intersection = len(set(y_pred[:k]) & set(y_true))
    return intersection / k

y_true = [1, 3, 5]
y_pred = [2, 1, 5, 4, 3]
print("Precision@3:", precision_at_k(y_true, y_pred, 3))  # 输出 2/3 ≈ 0.6667
```



### 2. 召回率@K（Recall@K）
**原理**：推荐列表中前K个物品覆盖的真实相关物品的比例。  
**公式**：
$$
Recall@K = \frac{\text{推荐的前K个物品中相关数}}{\text{用户总相关物品数}}
$$

**Python实现**：
```python
def recall_at_k(y_true, y_pred, k):
    intersection = len(set(y_pred[:k]) & set(y_true))
    return intersection / len(y_true) if len(y_true) > 0 else 0

print("Recall@3:", recall_at_k(y_true, y_pred, 3))  # 输出 2/3 ≈ 0.6667
```



### 3. 命中率@K（Hit Rate@K）
**原理**：推荐的Top-K列表中是否包含至少一个相关物品的比例。  
**公式**：
$$
HitRate@K = \frac{\text{命中次数}}{\text{总用户数}}
$$

**Python实现**：
```python
def hit_rate_at_k(y_true_list, y_pred_list, k):
    hits = 0
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        if len(set(y_pred[:k]) & set(y_true)) > 0:
            hits += 1
    return hits / len(y_true_list)

# 示例：多个用户的真实和预测列表
y_true_list = [[1,3], [2,5], [4]]
y_pred_list = [[2,1,5], [3,2,6], [4,7,8]]
print("HitRate@2:", hit_rate_at_k(y_true_list, y_pred_list, 2))  # 输出 2/3 ≈ 0.6667
```



### 4. 覆盖率（Coverage）
**原理**：推荐系统能够推荐的物品占全量物品的比例。  
**公式**：
$$
Coverage = \frac{|\cup_{u} \text{RecItems}_u|}{|I|}
$$
- $|I|$：全量物品数。

**Python实现**：
```python
def coverage(all_items, recommended_lists):
    recommended_items = set()
    for rec_list in recommended_lists:
        recommended_items.update(rec_list)
    return len(recommended_items) / len(all_items)

all_items = [1, 2, 3, 4, 5, 6, 7, 8]
recommended_lists = [[1,2,3], [2,4,5], [3,6,7]]
print("Coverage:", coverage(all_items, recommended_lists))  # 输出 6/8 = 0.75
```



### 5. 多样性（Diversity）
**原理**：衡量推荐列表中物品的差异性，常用相似度矩阵计算。  
**公式**（基于物品相似度）：
$$
Diversity = 1 - \frac{\sum_{i,j \in RecList} sim(i,j)}{K(K-1)/2}
$$
- $sim(i,j)$：物品i和j的相似度（如余弦相似度）。

**Python实现**：
```python
from sklearn.metrics.pairwise import cosine_similarity

def diversity(recommended_list, item_features):
    # item_features: 物品特征矩阵
    sim_matrix = cosine_similarity(item_features[recommended_list])
    n = len(recommended_list)
    if n < 2:
        return 1.0
    total_sim = np.sum(sim_matrix[np.triu_indices(n, k=1)])
    return 1 - total_sim / (n * (n-1) / 2)

# 示例：物品特征（如Embedding向量）
item_features = np.array([
    [0.9, 0.1, 0.2],
    [0.8, 0.3, 0.1],
    [0.1, 0.9, 0.4]
])
rec_list = [0, 1, 2]
print("Diversity:", diversity(rec_list, item_features))
```



### 6. 新颖性（Novelty）
**原理**：推荐物品的流行度逆权重，避免推荐热门物品。  
**公式**（基于信息熵）：
$$
Novelty = -\sum_{i \in RecList} p(i) \log p(i)
$$
- $p(i)$：物品i的流行度（出现次数 / 总推荐次数）。

**Python实现**：
```python
from collections import defaultdict

def novelty(recommended_lists):
    # 统计物品流行度
    item_counts = defaultdict(int)
    total = 0
    for rec_list in recommended_lists:
        for item in rec_list:
            item_counts[item] += 1
            total += 1
    # 计算熵
    entropy = 0
    for count in item_counts.values():
        p = count / total
        entropy -= p * np.log(p)
    return entropy

recommended_lists = [[1,2,3], [2,3,4], [3,4,5]]
print("Novelty:", novelty(recommended_lists))
```



## 三、业务指标（结合用户行为）

### 1. 点击率（CTR, Click-Through Rate）
**公式**：
$$
CTR = \frac{\text{点击次数}}{\text{曝光次数}}
$$

### 2. 转化率（CVR, Conversion Rate）
**公式**：
$$
CVR = \frac{\text{购买/转化次数}}{\text{点击次数}}
$$

### 3. 平均会话时长（Average Session Duration）
**公式**：
$$
\text{AvgSessionDuration} = \frac{\sum \text{会话时长}}{\text{会话数}}
$$



## 四、指标对比与选择建议
| 指标            | 适用场景                  | 优点                          | 缺点                      |
|----------------|-------------------------|-----------------------------|--------------------------|
| **MAP**        | 多查询排序任务            | 综合评估排序质量               | 计算复杂度高               |
| **NDCG**       | 多级相关性排序（如搜索）    | 考虑位置折扣和归一化            | 需要定义相关性分级           |
| **MRR**        | 强调首个相关结果的位置      | 简单直观                      | 忽略后续相关结果            |
| **Precision@K**| Top-K推荐效果评估          | 直接反映推荐准确性              | 不区分相关程度              |
| **Coverage**   | 评估推荐系统的长尾覆盖      | 避免推荐过于集中热门物品         | 无法衡量准确性              |
| **Diversity**  | 提升用户体验多样性          | 减少推荐重复内容                | 依赖物品相似度计算           |



## 五、总结
- **排序任务**：优先选择MAP、NDCG、MRR，需结合位置敏感性和相关性分级。  
- **推荐任务**：综合使用Precision@K、Recall@K、Coverage、Diversity，平衡准确性与多样性。  
- **业务场景**：结合CTR、CVR等在线指标验证模型的实际收益。  
- **多维度评估**：单一指标易导致偏差，需联合多个指标全面分析。