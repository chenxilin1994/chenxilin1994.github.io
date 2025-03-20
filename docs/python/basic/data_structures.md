---
layout: doc
title: Python数据结构详解
editLink: true
---

# Python核心数据结构

## 1. 列表(List)

### 特性
- 有序集合
- 可变类型
- 允许重复元素

### 基本操作
```python
# 创建列表
fruits = ['apple', 'banana', 'cherry']

# 增删改查
fruits.append('orange')        # 追加元素
fruits.insert(1, 'grape')      # 插入元素
fruits[2] = 'mango'            # 修改元素
del fruits[0]                  # 删除元素
```

### 常用方法
| 方法 | 说明 | 时间复杂度 |
|--|--|--|
| `append(x)` | 尾部添加元素 | O(1) |
| `pop([i])`  | 删除指定位置元素 | O(n) |
| `sort()`    | 原地排序     | O(n log n) |



## 2. 元组(Tuple)

### 特性
```python
# 创建元组
coordinates = (40.7128, -74.0060)
```

### 与列表对比
| 特性         | 列表 | 元组 |
|--|--|--|
| 可变性       | ✓    | ×    |
| 内存占用     | 较大 | 较小 |
| 适用场景     | 动态数据 | 固定数据 |



## 3. 字典(Dict)

### 基本使用
```python
# 创建字典
user = {
    "name": "Alice",
    "age": 30,
    "is_active": True
}

# 访问元素
print(user.get("email", "N/A"))  # 安全访问
```

### 字典推导式
```python
squares = {x: x**2 for x in range(5)}
# {0:0, 1:1, 2:4, 3:9, 4:16}
```



## 4. 集合(Set)

### 特性演示
```python
A = {1, 2, 3}
B = {3, 4, 5}

print(A | B)  # 并集 {1,2,3,4,5}
print(A & B)  # 交集 {3}
print(A - B)  # 差集 {1,2}
```

### 应用场景
- 去重：`unique = list(set(duplicates))`
- 成员测试：`if x in visited:`



## 5. 字符串(String)

### 常用操作
```python
text = "Python Programming"

# 字符串切片
print(text[7:18])    # "Programming"
# 方法调用
print(text.lower())  # "python programming"
```

### 格式化方法对比
```python
name = "Alice"
# f-string (Python3.6+)
print(f"Hello {name}")  
# format方法
print("Hello {}".format(name))
```

## 数据结构选择指南

| 需求场景                 | 推荐结构       |
|--|--|
| 维护元素添加顺序         | 列表/OrderedDict |
| 快速键值查找             | 字典           |
| 数据不可变               | 元组           |
| 元素唯一性               | 集合           |
| 文本处理                 | 字符串         |



## 性能优化技巧

1. **预分配列表空间**
```python
# 低效
lst = []
for i in range(10000):
    lst.append(i)

# 高效
lst = [0] * 10000
for i in range(10000):
    lst[i] = i
```

2. **字典代替多条件判断**
```python
# 常规写法
if status == 200:
    handle_success()
elif status == 404:
    handle_not_found()

# 优化写法
handlers = {
    200: handle_success,
    404: handle_not_found
}
handlers.get(status, default_handler)()
```

