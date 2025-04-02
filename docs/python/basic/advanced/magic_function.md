# 魔法函数

## 什么是魔法函数
python定义类时中，以双下划线开头，以双下划线结尾函数为魔法函数

- 魔法函数可以定义类的特性
- 魔法函数是解释器提供的功能
- 魔法函数只能使用 python 提供的魔法函数，不能自定义

```python
class Company:
    def __init__(self, employee_list):
        self.employee = employee_list

    def __getitem__(self, index):
        return self.employee[index]


company = Company(['alex', 'linda', 'catherine'])
employee = company.employee

for item in employee:
    print(item)

# for 首先去找 __iter__, 没有时优化去找__getitem__
for item in company:
    print(item)
```

## Python数据模型

数据模型，涉及到知识点其实就是魔法函数

- 魔法函数会影响 python 语法 company[:2]
- 魔法函数会影响内置函数调用 len(company)

## 魔法函数一览

### 非数据运算

字符串表示

- `__repr__`
- `__str__`

```python
class Company:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return '<Company [%s]>' % self.name

    def __repr__(self):
        return '<Company [%s]>' % self.name


company = Company('Apple')
print(company)

# Python 解释器会隐含调用
print(company.__repr__())
```

集合、序列相关

- `__len__`
- `__getitem__`
- `__setitem__`
- `__delitem__`
- `__contains__`

迭代相关

- `__iter__`
- `__next__`

可调用

- `__call__`

with 上下文管理器

- `__enter__`
- `__exit__`

### 数据运算

- `__abs__`
- `__add__`

```python
class Num:
    def __init__(self, num):
        self.num = num

    def __abs__(self):
        return abs(self.num)


n = Num(-1)
print(abs(n))


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __str__(self):
        return 'Vector(%s, %s)' % (self.x, self.y)


v1 = Vector(1, 3)
v2 = Vector(2, 4)
print(v1 + v2)
```

## len函数的特殊性

CPython时，向list，dict等内部做过优化，len(list)效率高

