# 深入类和对象

## 1. 鸭子类型和多态

### 1.1 鸭子类型
当看到一只鸟走起来像鸭子、游泳起来像鸭子、叫起来像鸭子，那么这只鸟可以被称为鸭子

```python
class Cat:
    def say(self):
        print('I am a cat.')

class Dog:
    def say(self):
        print('I am a dog.')

class Duck:
    def say(self):
        print('I am a duck.')


# Python中较灵活，只要实现say方法就行，实现了多态
animal = Cat
animal().say()

# 实现多态只要定义了相同方法即可
animal_list = [Cat, Dog, Duck]
for an in animal_list:
    an().say()

"""
class Animal:
    def say(self):
        print('I am a animal.')

# 需要继承Animal，并重写say方法
class Cat(Animal):
   def say(self):
       print('I am a cat.')

# Java 中定义需要指定类型
Animal an = new Cat()
an.say()
"""

li1 = ['i1', 'i2']
li2 = ['i3', 'i4']

tu = ('i5', 'i6')
s1 = set()
s1.add('i7')
s1.add('i8')

# 转变观念，传入的不单单是list，甚至自己实现 iterable 对象
li1.extend(li2)     # iterable
li1.extend(tu)
li1.extend(s1)
print(li1)
```

- 实现多态只要定义了相同方法即可
- 魔法函数充分利用了鸭子类型的特性，只要把函数塞进类型中即可

## 2. 抽象基类(abc模块)

### 2.1 抽象基类

- 抽象基类无法实例化
- 变量没有类型限制，可以指向任何类型
- 抽象基类和魔法函数构成了Python的基础，即协议

在抽象基类定义了抽象方法，继承抽象基类的类，必须实现这些方法

场景一：想判断某个对象的类型

```python {cmd="python3"}
class Company:
    def __init__(self, name):
        self.name = name

    def __len__(self):
        return len(self.name)

company = Company('Linda Process Ltd.')
print(hasattr(company, '__len__'))

from collections.abc import Sized
print(isinstance(company, Sized))
```

```
True
True
```

场景二：强制子类必须实现某些方法
```python {cmd="python3"}
import abc

class CacheBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get(self, key):
        pass

    @abc.abstractmethod
    def set(self, key, value):
        pass

class MemoryCache(CacheBase):
    pass

m = MemoryCache()
print(m)
```

```python
Traceback (most recent call last):
  File "c:\Users\xiligey\Alipan\website\docs\python\basic\advanced\f52qolj1m_code_chunk.python3", line 16, in <module>
    m = MemoryCache()
TypeError: Can't instantiate abstract class MemoryCache with abstract methods get, set
```

注意：抽象基类容易设计过度，多继承推荐使用Mixin

## 3. isinstance和type的区别
- isinstance会去查找继承链
- type只会判断变量的内存地址

```python {cmd="python3"}
class A:
    pass

class B(A):
    pass

b = B()
print(isinstance(b, B))
print(isinstance(b, A))

print(type(b) is B)
print(type(b) is A)
```
```python
True
True
True
False
```
## 4. 类变量和实例变量
- 类变量定义与使用
- 实例变量定义与使用
- 类变量是所有实例变量共享

```python {cmd="python3"}
class A:
    aa = 1

    def __init__(self, x, y):
        self.x = x
        self.y = y

a = A(2, 3)
b = A(4, 5)
print(a.x, a.y)

A.aa = 111
a.aa = 100  # 新建一个a的属性aa，100赋值给该aa
A.aa = 222
print(A.aa, a.aa, b.aa)
```
<!-- 
## 5. 类属性和实例属性以及查找顺序

- 类属性：定义在类中的变量和方法
- 实例属性：__init__中定义

### 深度优先 DFS


## 6. 静态方法、类方法、对象方法以及参数

## 7. 数据封装和私有属性

## 8. Python对象的自省机制

## 9. super真的是调用父类吗 -->