---
layout: doc
title: 数据结构基础：数组的全面解析与Python实现
editLink: true
---

# 数组：数据结构的基石

## 第一章 数组基础理论

### 1.1 数组的定义与特性
数组（Array）是一种**线性数据结构**，由相同类型元素的集合组成，存储在**连续的内存空间**中。其核心特征：

- **同构性**：所有元素类型相同
- **连续性**：内存地址连续分配
- **确定性**：创建时需指定容量（静态数组）
- **随机访问**：通过索引直接访问任意元素

### 1.2 内存结构解析
假设存储int类型数组（4字节/元素）：

```python
索引：   0       1       2        3
地址：0x1000 0x1004 0x1008 0x100C
值：   10     20     30      40
```

元素地址计算公式：  
`address[i] = base_address + i * sizeof(type)`

### 1.3 核心操作复杂度
| 操作            | 时间复杂度 | 说明                         |
|----------------|------------|------------------------------|
| 访问元素        | O(1)       | 直接计算内存地址              |
| 插入元素（末尾）| O(1)       | 动态数组的摊销时间复杂度      |
| 插入元素（中间）| O(n)       | 需要移动后续元素              |
| 删除元素        | O(n)       | 需要移动后续元素              |
| 查找元素        | O(n)       | 无序数组需要线性扫描          |


## 第二章 Python中的数组实现

### 2.1 列表的本质
Python的`list`实际上是**动态数组**实现，具有以下特性：

- 自动扩容/缩容机制
- 可存储异构数据（但推荐保持同构）
- 预分配额外空间减少频繁扩容

### 2.2 动态数组扩容策略
```python
import sys

def show_growth():
    lst = []
    last_capacity = 0
    for i in range(100):
        lst.append(i)
        curr_capacity = sys.getsizeof(lst) // sys.getsizeof(0)
        if curr_capacity != last_capacity:
            print(f"元素数: {i+1}, 容量: {curr_capacity}")
            last_capacity = curr_capacity

show_growth()

# 输出示例：
# 元素数: 1, 容量: 4
# 元素数: 5, 容量: 8
# 元素数: 9, 容量: 16
# 元素数: 17, 容量: 25
# 元素数: 26, 容量: 35...
```
扩容策略：新容量 = 当前容量 * ~1.125（CPython实现）

### 2.3 基础操作实现

#### 创建数组
```python
# 标准创建
arr = [1, 2, 3, 4, 5]

# 使用生成式
zeros = [0] * 10  # [0, 0, ..., 0]

# 类型限定（array模块）
import array
int_arr = array.array('i', [1, 2, 3])
```

#### 访问元素
```python
# 正索引
print(arr[0])  # 第一个元素 → 1

# 负索引
print(arr[-1]) # 最后一个元素 → 5

# 切片操作
print(arr[1:4]) # [2, 3, 4]
```

#### 插入操作
```python
# 末尾追加（O(1)）
arr.append(6) 

# 指定位置插入（O(n)）
arr.insert(2, 2.5)  # [1, 2, 2.5, 3, 4, 5, 6]
```

#### 删除操作
```python
# 按值删除（O(n)）
arr.remove(2.5) 

# 按索引删除（O(n)）
del arr[1]

# 弹出末尾（O(1)）
last = arr.pop()
```


## 第三章 高级操作与应用

### 3.1 内存视图
```python
# 创建数组
original = array.array('d', [1.0, 2.0, 3.0])

# 获取内存视图
mem_view = memoryview(original)

# 修改视图影响原数组
mem_view[1] = 4.0
print(original)  # array('d', [1.0, 4.0, 3.0])
```

### 3.2 数组旋转算法
实现将数组元素向右旋转k次：

```python
def rotate(nums, k):
    n = len(nums)
    k %= n
    nums[:] = nums[-k:] + nums[:-k]

# 示例
arr = [1,2,3,4,5]
rotate(arr, 2)
print(arr)  # [4,5,1,2,3]
```

### 3.3 多维数组实现
```python
# 二维数组（列表推导式）
matrix = [[0]*5 for _ in range(3)]

# 访问元素
matrix[0][2] = 10

# 锯齿数组
jagged = [
    [1,2,3],
    [4,5],
    [6,7,8,9]
]
```


## 第四章 性能优化实践

### 4.1 预分配空间
```python
# 低效方式
result = []
for i in range(10000):
    result.append(i)

# 高效方式
result = [0] * 10000
for i in range(10000):
    result[i] = i
```

### 4.2 批量操作
```python
# 低效
for item in data:
    lst.append(item)

# 高效
lst.extend(data)
```

### 4.3 内存优化对比
```python
from pympler import asizeof

lst = [i for i in range(1000)]
arr = array.array('i', lst)

print(f"列表内存: {asizeof.asizeof(lst)/1024:.2f} KB")
print(f"数组内存: {asizeof.asizeof(arr)/1024:.2f} KB")

# 典型输出：
# 列表内存: 36.12 KB
# 数组内存: 4.12 KB
```


## 第五章 典型应用场景

### 5.1 位图索引
```python
class Bitmap:
    def __init__(self, size):
        self.size = size
        self.bits = array.array('B', [0]*( (size+7)//8 ))

    def set_bit(self, pos):
        index = pos // 8
        offset = pos % 8
        self.bits[index] |= 1 << offset

    def test_bit(self, pos):
        index = pos // 8
        offset = pos % 8
        return (self.bits[index] & (1 << offset)) != 0
```

### 5.2 环形缓冲区
```python
class CircularBuffer:
    def __init__(self, capacity):
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0

    def enqueue(self, item):
        if self.size == len(self.buffer):
            self._expand()
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % len(self.buffer)
        self.size += 1

    def _expand(self):
        new_cap = len(self.buffer) * 2
        new_buffer = [None] * new_cap
        for i in range(self.size):
            new_buffer[i] = self.buffer[(self.head + i) % len(self.buffer)]
        self.buffer = new_buffer
        self.head = 0
        self.tail = self.size
```


## 练习题

1. 实现数组去重算法，要求时间复杂度O(n)，空间复杂度O(1)
2. 编写函数找出数组中消失的数字（LeetCode 448）
3. 实现两个有序数组的归并排序
4. 设计循环双端数组，支持O(1)时间的头尾插入/删除


### 附录：时间复杂度验证实验

```python
import timeit

def test_append():
    lst = []
    for i in range(100000):
        lst.append(i)

def test_insert():
    lst = []
    for i in range(1000):  # 数量级减少
        lst.insert(0, i)

print("尾部追加耗时:", timeit.timeit(test_append, number=100))
print("头部插入耗时:", timeit.timeit(test_insert, number=100))

# 典型输出：
# 尾部追加耗时: 0.89秒
# 头部插入耗时: 2.31秒 （虽然只操作1/100数据量）
``` 
