# Apriori算法



## 一、Apriori算法的核心思想
Apriori算法是关联规则挖掘中的经典方法，用于**发现事务数据库中的频繁项集**（即频繁共同出现的物品组合），并基于频繁项集生成**关联规则**（如“购买A则可能购买B”）。其核心思想基于以下两点：
1. **先验原理**：若一个项集是频繁的，则它的所有子集也一定是频繁的；反之，若某个子集不频繁，则其超集一定不频繁。
2. **逐层搜索**：通过逐层生成候选项集并剪枝（利用先验原理），减少计算量。



## 二、关键概念与公式
1. **项集（Itemset）**：一组物品的集合，如 \{牛奶, 面包\}。
2. **支持度（Support）**：项集在所有事务中出现的频率。
   $$
   \text{Support}(X) = \frac{\text{包含X的事务数}}{\text{总事务数}}
   $$
3. **置信度（Confidence）**：在包含X的事务中同时包含Y的条件概率。
   $$
   \text{Confidence}(X \to Y) = \frac{\text{Support}(X \cup Y)}{\text{Support}(X)}
   $$
4. **频繁项集**：支持度不低于预设阈值（`min_support`）的项集。
5. **关联规则**：形如 $X \to Y$ 的规则，要求置信度不低于阈值（`min_confidence`）。



## 三、算法步骤详解
1. **生成频繁1-项集（L₁）**  
   扫描事务数据库，统计所有单个物品的支持度，筛选出满足 `min_support` 的项集。

2. **逐层生成候选k-项集（Cₖ）**  
   通过连接步（Join）和剪枝步（Prune）生成候选k-项集：
   - **连接步**：将频繁(k-1)-项集（Lₖ₋₁）两两连接，生成候选k-项集。  
     例如：\{A,B\} 和 \{A,C\} 连接生成 \{A,B,C\}。
   - **剪枝步**：移除候选k-项集中存在(k-1)-子集不在Lₖ₋₁中的项集。

3. **筛选频繁k-项集（Lₖ）**  
   扫描数据库，计算候选k-项集的支持度，保留满足 `min_support` 的项集。

4. **重复步骤2-3**，直到无法生成更大的频繁项集。

5. **生成关联规则**  
   对每个频繁项集 $Z$，生成所有可能的非空子集 $X$ 和 $Y = Z \setminus X$，计算置信度，保留满足 `min_confidence` 的规则。



## 四、Python代码实现

以下代码实现Apriori算法，包含频繁项集挖掘和关联规则生成：

```python
import numpy as np
from itertools import combinations

def generate_candidates(itemsets, k):
    """生成候选k-项集（连接步）"""
    candidates = set()
    for i in range(len(itemsets)):
        for j in range(i+1, len(itemsets)):
            # 若前k-2项相同，则合并生成新项集
            if itemsets[i][:k-2] == itemsets[j][:k-2]:
                new_candidate = tuple(sorted(set(itemsets[i]).union(itemsets[j])))
                if len(new_candidate) == k:
                    candidates.add(new_candidate)
    return [list(c) for c in candidates]

def prune_candidates(candidates, prev_itemsets, k):
    """剪枝：移除包含非频繁(k-1)-子集的候选项（剪枝步）"""
    pruned = []
    for candidate in candidates:
        # 生成所有k-1子集
        subsets = list(combinations(candidate, k-1))
        # 检查所有子集是否在prev_itemsets中
        valid = True
        for subset in subsets:
            if list(subset) not in prev_itemsets:
                valid = False
                break
        if valid:
            pruned.append(candidate)
    return pruned

def apriori(transactions, min_support=0.5):
    """Apriori算法主函数"""
    # 初始化频繁1-项集
    items = sorted(set(item for transaction in transactions for item in transaction))
    item_counts = {tuple([item]):0 for item in items}
    for transaction in transactions:
        for item in items:
            if item in transaction:
                item_counts[tuple([item])] += 1
    n_transactions = len(transactions)
    L1 = [ [item] for item, count in item_counts.items() if count/n_transactions >= min_support ]
    L = [L1]
    k = 2
    
    # 逐层生成频繁项集
    while True:
        # 生成候选k-项集
        candidates = generate_candidates(L[-1], k)
        # 剪枝
        candidates = prune_candidates(candidates, L[-1], k)
        # 计算支持度
        candidate_counts = {tuple(c):0 for c in candidates}
        for transaction in transactions:
            for candidate in candidates:
                if set(candidate).issubset(set(transaction)):
                    candidate_counts[tuple(candidate)] += 1
        # 筛选满足min_support的项集
        Lk = [ list(c) for c, count in candidate_counts.items() if count/n_transactions >= min_support ]
        if not Lk:
            break
        L.append(Lk)
        k += 1
    
    # 合并所有频繁项集
    frequent_itemsets = []
    for itemsets in L:
        frequent_itemsets.extend(itemsets)
    return frequent_itemsets

def generate_rules(frequent_itemsets, transactions, min_confidence=0.7):
    """生成关联规则"""
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue
        # 生成所有非空子集作为前件
        all_subsets = []
        for i in range(1, len(itemset)):
            all_subsets.extend(combinations(itemset, i))
        for antecedent in all_subsets:
            antecedent = list(antecedent)
            consequent = list(set(itemset) - set(antecedent))
            # 计算支持度和置信度
            support_antecedent = sum(1 for t in transactions if set(antecedent).issubset(t)) / len(transactions)
            support_rule = sum(1 for t in transactions if set(itemset).issubset(t)) / len(transactions)
            confidence = support_rule / support_antecedent
            if confidence >= min_confidence:
                rules.append( (antecedent, consequent, confidence) )
    return rules

# 示例数据：购物篮事务
transactions = [
    ['牛奶', '面包', '啤酒'],
    ['牛奶', '尿布', '啤酒', '鸡蛋'],
    ['面包', '尿布', '啤酒', '可乐'],
    ['牛奶', '面包', '尿布', '啤酒'],
    ['牛奶', '面包', '尿布', '可乐']
]

# 运行Apriori算法
min_support = 0.4  # 最小支持度阈值
min_confidence = 0.6  # 最小置信度阈值
frequent_itemsets = apriori(transactions, min_support)
rules = generate_rules(frequent_itemsets, transactions, min_confidence)

# 输出结果
print("频繁项集：")
for itemset in frequent_itemsets:
    print(f"{itemset}: 支持度 = {sum(1 for t in transactions if set(itemset).issubset(t))/len(transactions):.2f}")

print("\n关联规则：")
for rule in rules:
    antecedent, consequent, confidence = rule
    print(f"{antecedent} => {consequent}: 置信度 = {confidence:.2f}")
```



## 五、代码解析
1. **generate_candidates**：通过连接步生成候选k-项集，确保新项集的(k-1)-前缀相同。
2. **prune_candidates**：利用先验原理剪枝，移除包含非频繁子集的候选项。
3. **apriori**：逐层生成频繁项集，直至无法扩展。
4. **generate_rules**：基于频繁项集生成关联规则，筛选满足置信度阈值的规则。



## 六、关键点说明
- **效率优化**：Apriori通过减少候选项集数量（剪枝）降低计算复杂度，但仍可能在大数据集上较慢。
- **参数选择**：`min_support`和`min_confidence`需根据数据特性调整，过高会导致规则过少，过低则生成大量无意义规则。
- **应用场景**：适用于超市购物篮分析、推荐系统、交叉销售等关联分析任务。



## 七、示例输出
```
频繁项集：
['牛奶']: 支持度 = 0.80
['面包']: 支持度 = 0.80
['啤酒']: 支持度 = 0.80
['尿布']: 支持度 = 0.80
['尿布', '牛奶']: 支持度 = 0.60
['尿布', '面包']: 支持度 = 0.60
...

关联规则：
['牛奶'] => ['尿布']: 置信度 = 0.75
['面包'] => ['尿布']: 置信度 = 0.75
['尿布'] => ['牛奶']: 置信度 = 0.75
...
```



通过上述代码，可以直观地看到Apriori算法如何从购物篮数据中挖掘出频繁购买的商品组合及关联规则。