---
layout: doc
title: Python基础语法精要
editLink: true
---

# Python基础语法

## 1. 基本结构

### 代码注释
```python
# 单行注释
"""
多行注释（本质是字符串）
"""
```

### 代码缩进

```python
if True:
    print("正确缩进")  # 必须使用4个空格
    # 同一代码块缩进必须一致
```


## 2. 变量与数据类型

### 变量定义
```python
counter = 100          # 整型
miles = 999.0          # 浮点型 
name = "John"          # 字符串
is_active = True       # 布尔型
```

### 主要数据类型
| 类型       | 示例                   | 说明                |
|------------|------------------------|---------------------|
| `int`      | `x = 10`              | 整数                |
| `float`    | `y = 3.14`            | 浮点数              |
| `str`      | `s = "Hello"`         | 字符串              |
| `bool`     | `flag = True`         | 布尔值              |
| `list`     | `lst = [1, 2, 3]`     | 列表（可修改）      |
| `tuple`    | `tup = (1, "a")`      | 元组（不可修改）    |
| `dict`     | `d = {"name": "John"}`| 字典（键值对）      |

## 3. 运算符

### 算术运算
```python
print(10 + 3)   # 13
print(10 - 3)   # 7
print(10 * 3)   # 30
print(10 ** 3)  # 1000（幂运算）
print(10 / 3)   # 3.333...
print(10 // 3)  # 3（整除）
```

### 比较运算
```python
print(3 == 3.0)  # True（值相等）
print(3 is 3.0)  # False（对象不同）
print(3 != 2)    # True
```


## 4. 流程控制

### 条件语句
```python
age = 18
if age < 0:
    print("无效年龄")
elif age < 18:
    print("未成年人")
else:
    print("成年人")
```

### 循环结构
```python
# while循环
count = 0
while count < 5:
    print(count)
    count += 1

# for循环
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)
```


## 5. 函数基础

### 函数定义
```python
def greet(name):
    """返回问候语"""
    return f"Hello, {name}!"

print(greet("Alice"))  # Hello, Alice!
```

### 参数传递
```python
def update_list(lst):
    lst.append(4)

my_list = [1, 2, 3]
update_list(my_list)
print(my_list)  # [1, 2, 3, 4]
```


## 6. 输入输出

### 控制台交互
```python
name = input("请输入姓名：")
print(f"欢迎您，{name}！")

# 格式化输出
print("价格：%.2f" % 99.987)  # 价格：99.99
print(f"结果：{10*10}")       # 结果：100
```


## 常见错误示例

❌ 错误缩进
```python
if True:
print("缺少缩进")  # IndentationError
```

❌ 变量未定义
```python
print(undefined_var)  # NameError
```

❌ 修改元组
```python
t = (1, 2)
t[0] = 3  # TypeError
```
