---
layout: doc
title: Python函数编程详解
editLink: true
---

# 函数编程精要

## 1. 函数定义基础

### 基本语法
```python
def calculate_area(width, height):
    """计算矩形面积"""
    return width * height

print(calculate_area(5, 3))  # 输出：15
```

### 返回多个值
```python
def analyze_numbers(numbers):
    return min(numbers), max(numbers), sum(numbers)/len(numbers)

min_val, max_val, avg_val = analyze_numbers([2,5,7,1])
```


## 2. 参数传递机制

### 参数类型对比
| 参数类型       | 示例                      | 说明                     |
|----------------|---------------------------|--------------------------|
| 位置参数       | `func(a, b)`             | 按顺序传递               |
| 关键字参数     | `func(b=3, a=1)`         | 明确指定参数名           |
| 默认参数       | `def func(a=0)`          | 参数默认值               |
| 可变参数       | `def func(*args)`        | 接收元组                 |
| 关键字可变参数 | `def func(**kwargs)`     | 接收字典                 |

### 参数组合示例
```python
def complex_func(a, b=0, *args, **kwargs):
    print(f"a={a}, b={b}")
    print("可变参数:", args)
    print("关键字参数:", kwargs)

complex_func(1, 2, 3, 4, key1='v1', key2='v2')
```


## 3. 作用域规则

### LEGB规则
```python
x = 10  # Global

def outer():
    y = 20  # Enclosing
    def inner():
        z = 30  # Local
        print(x + y + z)
    inner()

outer()  # 输出：60
```

### global与nonlocal
```python
counter = 0

def increment():
    global counter
    counter += 1

def outer():
    count = 0
    def inner():
        nonlocal count
        count += 1
    inner()
    print(count)  # 输出：1
```


## 4. Lambda函数

### 使用场景
```python
# 简单运算
square = lambda x: x**2

# 排序辅助
users = [{'name':'Alice','age':25}, {'name':'Bob','age':30}]
users.sort(key=lambda x: x['age'])
```

### 限制说明
- 只能包含单个表达式
- 没有语句（if表达式可用）
- 不适合复杂逻辑


## 5. 装饰器原理

### 基础实现
```python
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"调用函数：{func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@logger
def add(a, b):
    return a + b

print(add(2,3))  # 先打印日志再返回结果
```

### 带参数的装饰器
```python
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello {name}!")

greet("Alice")  # 输出3次问候语
```


## 常见错误排查

❌ 修改不可变默认参数
```python
def append_to(num, target=[]):  # 危险！
    target.append(num)
    return target

print(append_to(1))  # [1]
print(append_to(2))  # [1,2] （非预期结果）
```

✅ 正确方式
```python
def append_to(num, target=None):
    if target is None:
        target = []
    target.append(num)
    return target
```


## 最佳实践建议

1. **单一职责原则**  
   每个函数只完成一个明确的任务

2. **合理控制函数长度**  
   建议不超过50行代码

3. **使用类型注解**（Python3.5+）
```python
from typing import List, Tuple

def process_data(data: List[int]) -> Tuple[float, float]:
    ...
```