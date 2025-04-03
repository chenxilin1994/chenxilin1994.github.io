

# 数据结构

## 11. 列表（list）和元组（tuple）的主要区别是什么？在什么场景下应优先使用元组？
答案  
- 核心区别：  
  1. 可变性：列表可变（支持增删改），元组不可变（创建后内容不可修改）。  
  2. 性能：元组的内存占用更小，创建和访问速度更快（适合存储常量数据）。  
  3. 哈希性：元组可作为字典的键（因其不可变性），列表不可。  

- 使用场景：  
  - 元组：存储不可变数据（如配置参数、数据库查询结果）、作为字典的键、函数返回多个值。  
  - 列表：需要动态修改数据的场景（如数据采集、缓存）。  

代码示例  
```python
# 性能对比
import sys
import timeit

lst = [1, 2, 3]
tpl = (1, 2, 3)
print(sys.getsizeof(lst))  # 88（64位Python）
print(sys.getsizeof(tpl))  # 72

# 时间测试
print(timeit.timeit('[1,2,3]'))        # 约0.04秒
print(timeit.timeit('(1,2,3)'))        # 约0.01秒

# 元组作为字典键
coordinates = {(35.7, 139.7): "Tokyo"}
```



## 12. 字典的键（key）需要满足什么条件？自定义对象如何作为字典的键？
答案  
- 键的条件：  
  1. 不可变性：键必须是不可变对象（如 `int`, `str`, `tuple`）。  
  2. 可哈希性：对象必须实现 `__hash__()` 和 `__eq__()` 方法。  

- 自定义对象作为键：  
  - 重写 `__hash__()` 和 `__eq__()` 方法，确保对象的哈希值在生命周期内不变。  
  - 若对象属性可变，需谨慎设计（修改属性可能导致哈希值变化，破坏字典一致性）。  

代码示例  
```python
class User:
    def __init__(self, id, name):
        self.id = id   # 不可变属性
        self.name = name

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

user1 = User(1, "Alice")
user2 = User(1, "Bob")
d = {user1: "data"}
print(d.get(user2))  # 输出 "data"（因id相同）
```



## 13. 解释列表推导式（List Comprehension）和生成器表达式（Generator Expression）的区别。
答案  
- 列表推导式：  
  - 立即生成完整的列表，占用内存存储所有元素。  
  - 语法：`[x for x in iterable if condition]`  
  - 适用场景：数据量小且需多次访问。  

- 生成器表达式：  
  - 惰性求值，逐个生成元素，节省内存。  
  - 语法：`(x for x in iterable if condition)`  
  - 适用场景：大数据量或只需单次遍历（如 `sum()` 的参数）。  

代码示例  
```python
# 内存占用对比
import sys
lst = [x for x in range(100000)]
gen = (x for x in range(100000))
print(sys.getsizeof(lst))  # 约824440字节
print(sys.getsizeof(gen))  # 约112字节

# 生成器用于惰性计算
total = sum(x * 2 for x in range(100000) if x % 2 == 0)
```



## 14. 如何合并两个字典？Python 3.9+ 提供了什么新特性？
答案  
- 传统方法：  
  1. `dict.update()`：原地修改第一个字典。  
  2. 字典解包：`{d1, d2}`（Python 3.5+）。  

- Python 3.9+ 新特性：  
  - `|` 运算符合并字典：`d3 = d1 | d2`（不修改原字典）。  
  - `|=` 运算符更新字典：`d1 |= d2`（原地修改 `d1`）。  

代码示例  
```python
d1 = {"a": 1, "b": 2}
d2 = {"b": 3, "c": 4}

# 方法1：update（修改d1）
d1.update(d2)
print(d1)  # {'a': 1, 'b': 3, 'c': 4}

# 方法2：解包（Python 3.5+）
d3 = {d1, d2}  # 注意键覆盖顺序

# 方法3：Python 3.9+
d4 = d1 | d2       # 合并为新字典
d1 |= d2            # 等价于d1.update(d2)
```



## 15. 集合（set）的主要应用场景有哪些？如何高效实现去重？
答案  
- 应用场景：  
  1. 去重：自动去除重复元素。  
  2. 成员检查：`O(1)` 时间复杂度（基于哈希表）。  
  3. 集合运算：交集（`&`）、并集（`|`）、差集（`-`）。  

- 高效去重：  
  - 直接转换列表为集合：`unique = list(set(lst))`（但会丢失顺序）。  
  - 保留顺序的去重：遍历列表并利用集合检查。  

代码示例  
```python
# 去重（不保留顺序）
lst = [3, 2, 2, 1, 3]
unique = list(set(lst))  # [1, 2, 3]

# 去重（保留顺序）
seen = set()
result = []
for item in lst:
    if item not in seen:
        seen.add(item)
        result.append(item)
print(result)  # [3, 2, 1]
```



## 16. 字符串的不可变性对性能有何影响？如何高效拼接多个字符串？
答案  
- 性能影响：  
  - 每次拼接字符串会创建新对象，时间复杂度为 `O(n^2)`（如 `s += "x"` 循环）。  

- 高效拼接方法：  
  1. `str.join()`：一次性拼接可迭代对象（时间复杂度 `O(n)`）。  
  2. `io.StringIO`：内存中动态构建字符串（类似列表追加）。  

代码示例  
```python
# 低效方法（避免使用）
s = ""
for _ in range(10000):
    s += "x"

# 高效方法1：join
parts = ["x" for _ in range(10000)]
s = "".join(parts)

# 高效方法2：StringIO
from io import StringIO
buffer = StringIO()
for _ in range(10000):
    buffer.write("x")
s = buffer.getvalue()
```



## 17. 生成器（Generator）的作用是什么？如何创建生成器？
答案  
- 作用：  
  - 惰性生成数据，节省内存（适用于大数据流或无限序列）。  
  - 通过 `yield` 关键字暂停函数执行，保留状态。  

- 创建方法：  
  1. 生成器函数：使用 `yield` 返回值的函数。  
  2. 生成器表达式：类似列表推导式，用 `()` 包裹。  

代码示例  
```python
# 生成器函数
def count_up_to(max):
    count = 1
    while count <= max:
        yield count
        count += 1

gen = count_up_to(5)
print(list(gen))  # [1, 2, 3, 4, 5]

# 生成器表达式
squares = (x2 for x in range(10))
print(next(squares))  # 0
```



## 18. 装饰器（Decorator）的原理是什么？如何编写一个保留函数元信息的装饰器？
答案  
- 原理：  
  - 装饰器是一个高阶函数，接收函数作为参数，返回新函数。  
  - 通过闭包和 `@` 语法糖实现功能增强（如日志、计时）。  

- 保留元信息：  
  - 使用 `functools.wraps` 保留原函数的 `__name__`、`__doc__` 等属性。  

代码示例  
```python
from functools import wraps

def logger(func):
    @wraps(func)
    def wrapper(*args, kwargs):
        print(f"调用函数：{func.__name__}")
        return func(*args, kwargs)
    return wrapper

@logger
def add(a, b):
    """返回两数之和"""
    return a + b

print(add.__name__)  # "add"（无wraps则为"wrapper"）
print(add(2, 3))     # 输出日志并返回5
```



## 19. 多线程和多进程在Python中的主要区别是什么？
答案  
- 多线程：  
  - 共享同一进程的内存空间，受GIL限制，适合I/O密集型任务。  
  - 线程间通信简单（如队列），但需处理线程安全问题。  

- 多进程：  
  - 每个进程有独立内存空间，绕过GIL，适合CPU密集型任务。  
  - 进程间通信复杂（需 `multiprocessing.Queue` 或管道）。  

代码示例  
```python
import threading
import multiprocessing

def task():
    print("执行任务")

# 多线程
thread = threading.Thread(target=task)
thread.start()

# 多进程
process = multiprocessing.Process(target=task)
process.start()
```



## 20. 如何实现深拷贝（Deep Copy）？什么情况下必须使用深拷贝？
答案  
- 实现方法：  
  - 使用 `copy.deepcopy()` 递归复制所有嵌套对象。  

- 使用场景：  
  1. 对象包含嵌套的可变结构（如列表中的列表）。  
  2. 需要完全独立的副本（修改副本不影响原对象）。  

代码示例  
```python
import copy

original = [[1, 2], {"a": 3}]
deep_copied = copy.deepcopy(original)

original[0].append(3)
original[1]["a"] = 4
print(deep_copied)  # [[1,2], {'a':3}]（完全独立）
```
