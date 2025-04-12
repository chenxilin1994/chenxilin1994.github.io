# 二分类评价指标

## 一、混淆矩阵（Confusion Matrix）
定义：分类结果的统计表，包含以下四个关键指标：
- TP（True Positive）：真实为正类，预测也为正类的样本数。
- TN（True Negative）：真实为负类，预测也为负类的样本数。
- FP（False Positive）：真实为负类，预测错误为正类的样本数（误报）。
- FN（False Negative）：真实为正类，预测错误为负类的样本数（漏报）。

|                | 预测为正类 | 预测为负类 |
|----------------|------------|------------|
| 真实为正类 | TP         | FN         |
| 真实为负类 | FP         | TN         |



## 二、核心评价指标

### 1. 准确率（Accuracy）
原理：所有预测正确的样本占总样本的比例。  
公式：
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$
适用场景：类别平衡的数据集。  
缺点：类别不平衡时可能误导（例如负类占99%，全预测负类准确率可达99%）。

Python实现：
```python
from sklearn.metrics import accuracy_score
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
print("Accuracy:", accuracy_score(y_true, y_pred))  # 输出 0.6
```



### 2. 精确率（Precision）
原理：预测为正类的样本中，真实为正类的比例。  
公式：
$$
\text{Precision} = \frac{TP}{TP + FP}
$$
适用场景：关注减少误报（如垃圾邮件检测，避免将正常邮件误判为垃圾）。

Python实现：
```python
from sklearn.metrics import precision_score
print("Precision:", precision_score(y_true, y_pred))  # 输出 0.6667
```



### 3. 召回率（Recall / Sensitivity / TPR）
原理：真实为正类的样本中，被正确预测为正类的比例。  
公式：
$$
\text{Recall} = \frac{TP}{TP + FN}
$$
适用场景：关注减少漏报（如癌症筛查，避免漏诊）。

Python实现：
```python
from sklearn.metrics import recall_score
print("Recall:", recall_score(y_true, y_pred))  # 输出 0.6667
```



### 4. F1分数（F1-Score）
原理：精确率和召回率的调和平均数，平衡二者关系。  
公式：
$$
\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$
适用场景：类别不平衡时综合评价模型性能。

Python实现：
```python
from sklearn.metrics import f1_score
print("F1 Score:", f1_score(y_true, y_pred))  # 输出 0.6667
```



### 5. 特异性（Specificity / TNR）
原理：真实为负类的样本中，被正确预测为负类的比例。  
公式：
$$
\text{Specificity} = \frac{TN}{TN + FP}
$$
适用场景：关注负类预测的可靠性（如疾病检测中的健康人判定）。

Python实现：
```python
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp)
print("Specificity:", specificity)  # 输出 0.5
```



### 6. ROC曲线与AUC值

#### 一、ROC曲线原理
ROC（Receiver Operating Characteristic）曲线通过展示不同分类阈值下的真阳性率（TPR）和假阳性率（FPR），评估二分类模型的性能。曲线的两个核心指标：
- TPR（True Positive Rate）：真实正例中被正确预测的比例，即召回率。
  $$
  TPR = \frac{TP}{TP + FN}
  $$
- FPR（False Positive Rate）：真实负例中被错误预测的比例。
  $$
  FPR = \frac{FP}{FP + TN}
  $$

#### 二、绘制步骤
以下通过示例数据和代码逐步解释绘制过程：

1. 准备数据
假设模型对5个样本的预测概率和真实标签如下：

| 样本索引 | 预测概率 | 真实标签 |
|----------|----------|----------|
| 1        | 0.9      | 1        |
| 2        | 0.8      | 0        |
| 3        | 0.7      | 1        |
| 4        | 0.7      | 0        |
| 5        | 0.6      | 1        |


2. 按预测概率降序排序样本
排序后：

| 样本索引 | 预测概率 | 真实标签 |
|----------|----------|----------|
| 1        | 0.9      | 1        |
| 2        | 0.8      | 0        |
| 3        | 0.7      | 1        |
| 4        | 0.7      | 0        |
| 5        | 0.6      | 1        |

3. 初始化参数
- 总正例数（P）：3（样本1、3、5）
- 总负例数（N）：2（样本2、4）
- 初始累积TP和FP：TP=0，FP=0

4. 遍历唯一阈值并计算TPR/FPR
遍历每个唯一预测概率作为阈值（降序处理）：

(a) 阈值 = 0.9
- 预测为正例的样本：样本1（概率≥0.9）
  - TP +=1 → TP=1
  - FP +=0 → FP=0
- 计算指标：
  - TPR = 1/(1+2) ≈ 0.333
  - FPR = 0/(0+2) = 0.0
- 当前点：(0.0, 0.333)

(b) 阈值 = 0.8
- 新增预测为正例的样本：样本2（概率≥0.8）
  - TP +=0 → TP=1
  - FP +=1 → FP=1
- 计算指标：
  - TPR = 1/3 ≈ 0.333
  - FPR = 1/(1+1) = 0.5
- 当前点：(0.5, 0.333)

(c) 阈值 = 0.7
- 新增预测为正例的样本：样本3、4（概率≥0.7）
  - TP +=1 → TP=2
  - FP +=1 → FP=2
- 计算指标：
  - TPR = 2/3 ≈ 0.666
  - FPR = 2/(2+0) = 1.0
- 当前点：(1.0, 0.666)

(d) 阈值 = 0.6
- 新增预测为正例的样本：样本5（概率≥0.6）
  - TP +=1 → TP=3
  - FP +=0 → FP=2
- 计算指标：
  - TPR = 3/3 = 1.0
  - FPR = 2/(2+0) = 1.0
- 当前点：(1.0, 1.0)

5. 添加起始点和结束点
- 起始点：(0.0, 0.0) —— 阈值为无穷大时，无预测为正例。
- 结束点：(1.0, 1.0) —— 阈值为0时，全预测为正例。

6. 连接所有点并绘制曲线
最终点序列：
```
(0.0, 0.0) → (0.0, 0.333) → (0.5, 0.333) → (1.0, 0.666) → (1.0, 1.0)
```

#### 三、Python代码实现
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 示例数据
y_true = np.array([1, 0, 1, 0, 1])
y_scores = np.array([0.9, 0.8, 0.7, 0.7, 0.6])

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# 绘制曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

#### 四、关键细节说明
1. 唯一阈值处理：合并相同预测概率的样本，避免重复计算。
2. 累积统计：逐步更新TP和FP，而非重新计算。
3. 除零问题：当TN=0时，FPR定义为1；当TP+FN=0时，TPR为0。


#### 五、总结
- ROC曲线直观性：通过曲线形状和AUC值快速判断模型性能。
- AUC解释：AUC=0.5表示随机猜测，AUC=1表示完美分类。
- 应用场景：适用于类别不平衡问题，且不受分类阈值影响。





### 7. PR曲线（Precision-Recall Curve）
原理：展示不同阈值下精确率与召回率的关系，适合类别不平衡数据。  
AP（Average Precision）：PR曲线下面积。

Python实现：
```python
from sklearn.metrics import precision_recall_curve, auc

precision, recall, _ = precision_recall_curve(y_true, y_probs)
ap = auc(recall, precision)

plt.plot(recall, precision, label=f'AP = {ap:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()
```



### 8. 马修斯相关系数（MCC）
原理：综合所有混淆矩阵元素的平衡指标，范围[-1, 1]，1表示完美预测。  
公式：
$$
\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$
适用场景：类别不平衡时的综合评估。

Python实现：
```python
from sklearn.metrics import matthews_corrcoef
print("MCC:", matthews_corrcoef(y_true, y_pred))  # 输出 0.258
```



## 三、手动计算示例
假设某模型的混淆矩阵如下：

|       | 预测为1 | 预测为0 |
|-------|---------|---------|
| 真实1 | TP=50   | FN=10   |
| 真实0 | FP=5    | TN=35   |

- `准确率 = (50+35)/(50+35+5+10) = 85/100 = 0.85`
- `精确率 = 50/(50+5) = 0.909`  
- `召回率 = 50/(50+10) = 0.833`  
- `F1 = 2*(0.909*0.833)/(0.909+0.833) ≈ 0.869 ` 
- `特异性 = 35/(35+5) = 0.875`
- `MCC = (50*35 - 5*10)/√[(50+5)(50+10)(35+5)(35+10)] ≈ 0.72`


## 四、总结与选择建议
1. 类别平衡时：优先使用准确率、F1、ROC-AUC。
2. 类别不平衡时：优先使用F1、PR曲线、MCC。
3. 减少误报：关注精确率。
4. 减少漏报：关注召回率。
5. 综合评估：ROC-AUC和MCC。

通过结合业务需求和数据分布选择合适的指标，避免单一指标的片面性。