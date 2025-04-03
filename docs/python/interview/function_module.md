
# 函数与模块

## 21. 默认参数在Python中为什么应该避免使用可变对象？如何正确设计默认参数？
详细答案  
- 问题原因：  
  Python的默认参数在函数定义时仅计算一次，而非每次调用时重新初始化。若默认参数是可变对象（如列表、字典），多次调用可能共享同一对象，导致意外修改。  

- 正确设计：  
  - 使用不可变对象（如 `None`）作为默认值，在函数内部初始化可变对象。  

代码示例  
```python
# 错误示例
def append_to(element, target=[]):
    target.append(element)
    return target

print(append_to(1))  # [1]
print(append_to(2))  # [1, 2]（默认列表被共享）

# 正确设计
def append_to_fixed(element, target=None):
    if target is None:
        target = []
    target.append(element)
    return target

print(append_to_fixed(1))  # [1]
print(append_to_fixed(2))  # [2]
```



## 22. Python的函数参数传递是“传值”还是“传引用”？如何理解？
详细答案  
- 传递机制：Python采用 “传对象引用”（Call by Object Reference）。  
  - 不可变对象（如整数、字符串）：函数内修改会创建新对象，不影响原变量。  
  - 可变对象（如列表、字典）：函数内直接修改会影响原对象。  

代码示例  
```python
def modify(num, lst):
    num += 1       # 创建新整数对象
    lst.append(4)  # 直接修改原列表

n = 10
l = [1, 2, 3]
modify(n, l)
print(n)  # 10（未改变）
print(l)  # [1, 2, 3, 4]（已修改）
```



## 23. 解释 `nonlocal` 和 `global` 关键字的区别。
详细答案  
- `global`：声明变量为全局作用域（模块级）。  
- `nonlocal`：声明变量为外层嵌套函数作用域（非全局）。  
- 使用场景：  
  - `global` 用于在函数内修改全局变量。  
  - `nonlocal` 用于在嵌套函数内修改外层函数的变量。  

代码示例  
```python
x = 0
def outer():
    y = 1
    def inner():
        global x
        nonlocal y
        x += 10
        y += 10
    inner()
    print(y)  # 11

outer()
print(x)  # 10
```



## 24. lambda函数有什么限制？适用哪些场景？
详细答案  
- 限制：  
  1. 只能包含单个表达式（不能有语句如 `return`、`if-else` 需用三元表达式）。  
  2. 无类型注解和文档字符串（可读性差）。  

- 适用场景：  
  - 简单匿名函数（如排序键 `sorted(lst, key=lambda x: x[1])`）。  
  - 函数式编程工具（如 `map`、`filter`）的参数。  

代码示例  
```python
# 三元表达式在lambda中的使用
max_value = lambda a, b: a if a > b else b
print(max_value(5, 3))  # 5

# 与map结合使用
squares = list(map(lambda x: x2, [1, 2, 3]))  # [1, 4, 9]
```



## 25. 什么是闭包（Closure）？如何判断一个函数是否是闭包？
详细答案  
- 闭包定义：  
  - 内部函数引用了外部函数的变量，且外部函数已执行完毕。  
  - 闭包保留了外层变量的状态（即使外层函数已退出）。  

- 判断方法：  
  - 检查函数的 `__closure__` 属性，若为 `None` 则不是闭包，否则返回包含变量的单元格（cell）。  

代码示例  
```python
def outer(x):
    def inner(y):
        return x + y
    return inner

closure_func = outer(10)
print(closure_func.__closure__)        # 输出单元格对象
print(closure_func.__closure__[0].cell_contents)  # 10
```



## 26. `__init__.py` 文件的作用是什么？Python 3.3+ 后还需要它吗？
详细答案  
- 传统作用：  
  - 标识目录为Python包（Package）。  
  - 可包含包初始化代码或定义 `__all__` 列表（控制 `from package import *` 的行为）。  

- Python 3.3+：  
  - 引入 命名空间包（Namespace Packages），允许无 `__init__.py` 的包。  
  - 但普通包仍需 `__init__.py` 以支持传统功能（如初始化代码）。  

代码示例  
```python
# 包结构示例
my_package/
├── __init__.py     # 定义 from .module import func
├── module.py
```



## 27. 如何动态导入模块（importlib的使用）？
详细答案  
- 应用场景：  
  - 按需加载模块（减少启动时间）。  
  - 插件化架构中动态加载功能。  

- 方法：  
  使用 `importlib.import_module()` 动态导入。  

代码示例  
```python
import importlib

# 动态导入os模块
module_name = "os"
os_module = importlib.import_module(module_name)
print(os_module.getcwd())  # 输出当前工作目录
```



## 28. Python的递归深度限制是什么？如何修改？
详细答案  
- 默认限制：  
  CPython的默认递归深度限制为 `1000`（可通过 `sys.getrecursionlimit()` 查看）。  

- 修改方法：  
  - 使用 `sys.setrecursionlimit(new_limit)`，但可能引发栈溢出风险。  

代码示例  
```python
import sys

print(sys.getrecursionlimit())  # 默认1000
sys.setrecursionlimit(1500)     # 修改限制（谨慎操作！）

def recursive(n):
    if n == 0:
        return
    recursive(n - 1)

recursive(1200)  # 若未修改限制会报错
```



## 29. 生成器（Generator）与协程（Coroutine）的区别是什么？
详细答案  
- 生成器：  
  - 用于生成数据序列（通过 `yield` 产生值）。  
  - 单向通信（调用者获取值，生成器无输入）。  

- 协程：  
  - 通过 `yield` 接收和发送数据（双向通信）。  
  - 用于并发编程（如异步IO），可用 `async def` 和 `await` 定义（Python 3.5+）。  

代码示例  
```python
# 生成器（单向）
def count_up_to(n):
    i = 0
    while i < n:
        yield i
        i += 1

# 协程（双向）
def coroutine():
    x = yield
    while True:
        x = yield x * 2

c = coroutine()
next(c)        # 启动协程
print(c.send(5))  # 输出10
```



## 30. 解释Python的内存管理机制（引用计数与垃圾回收）。
详细答案  
- 引用计数：  
  - 每个对象维护一个计数器，记录引用次数。当计数归零时立即释放内存。  
  - 优点：实时性高。缺点：无法处理循环引用。  

- 垃圾回收（GC）：  
  - 分代回收（Generational GC）解决循环引用问题。  
  - 对象分为三代（0代最年轻），年轻代对象更频繁检查。  

代码示例  
```python
import gc

# 循环引用示例
class Node:
    def __init__(self):
        self.next = None

a = Node()
b = Node()
a.next = b
b.next = a  # 循环引用

# 手动触发垃圾回收
gc.collect()  # 释放无法访问的循环引用对象
```
