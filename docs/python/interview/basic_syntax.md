# 基础语法

## 1. Python中可变对象和不可变对象的区别是什么？举例说明。
答案  
- 不可变对象（Immutable Objects）：对象一旦创建，其内容（值）不可修改。若尝试修改，Python会创建一个新对象。  
  - 类型：`int`, `float`, `str`, `tuple`, `frozenset`, `bytes`。  
  - 原理：不可变对象的内存地址在创建时固定，修改操作会触发新内存分配。  
  - 应用场景：适合作为字典的键（因为哈希值不变）。  

- 可变对象（Mutable Objects）：对象创建后，内容可被修改，内存地址不变。  
  - 类型：`list`, `dict`, `set`, `bytearray`。  
  - 原理：对象内部状态可变，修改操作直接在原内存地址进行。  
  - 应用场景：动态数据存储（如列表追加元素）。

代码示例  
```python
# 不可变对象示例（字符串）
s1 = "hello"
print(f"初始地址: {id(s1)}")  # 输出地址A
s1 += " world"                # 创建新对象
print(f"修改后地址: {id(s1)}") # 地址B ≠ 地址A

# 可变对象示例（列表）
lst = [1, 2]
print(f"初始地址: {id(lst)}")  # 输出地址C
lst.append(3)                 # 直接修改原对象
print(f"修改后地址: {id(lst)}") # 地址C不变
```



## 2. `is` 和 `==` 的区别是什么？
答案  
- `==`：值相等性检查（Value Equality）。  
  - 通过调用对象的 `__eq__()` 方法实现。  
  - 即使对象不同，只要内容相同即返回 `True`。  

- `is`：对象同一性检查（Identity Comparison）。  
  - 直接比较对象的内存地址（即 `id(a) == id(b)`）。  
  - 仅在两个变量引用同一对象时返回 `True`。  

关键场景  
- 小整数缓存（-5~256）和字符串驻留（Interned Strings）可能导致 `is` 返回 `True`，但不可依赖此特性。  

代码示例  
```python
a = 256
b = 256
print(a is b)   # True（小整数缓存）

c = 257
d = 257
print(c is d)   # 可能为False（非缓存范围）

x = [1, 2]
y = [1, 2]
print(x == y)   # True（值相等）
print(x is y)   # False（不同对象）
```



## 3. Python的深浅拷贝（Shallow Copy vs Deep Copy）有什么区别？
答案  
- 浅拷贝（Shallow Copy）：  
  - 仅复制顶层对象，嵌套对象仍引用原对象。  
  - 方法：`copy.copy()`、列表的 `list.copy()`、切片 `list[:]`。  
  - 问题：嵌套对象修改会影响浅拷贝结果。  

- 深拷贝（Deep Copy）：  
  - 递归复制所有嵌套对象，完全独立于原对象。  
  - 方法：`copy.deepcopy()`。  
  - 应用：需要完全隔离原对象修改的场景（如配置副本）。  

代码示例  
```python
import copy

original = [1, [2, 3], {"a": 4}]
shallow = copy.copy(original)
deep = copy.deepcopy(original)

# 修改嵌套对象
original[1].append(5)
original[2]["a"] = 10

print(shallow)  # [1, [2, 3, 5], {'a': 10}]（受影响）
print(deep)     # [1, [2, 3], {'a': 4}]     （独立）
```



## 4. 解释Python的GIL（全局解释器锁）及其影响。
答案  
- GIL（Global Interpreter Lock）：  
  - CPython解释器的线程同步机制，同一时刻仅允许一个线程执行Python字节码。  
  - 目的：简化CPython内存管理（避免并发竞争）。  

- 影响：  
  - CPU密集型任务：多线程无法利用多核，性能甚至可能下降（线程切换开销）。  
  - I/O密集型任务：线程在I/O等待时会释放GIL，多线程仍有效。  

- 解决方案：  
  - 使用多进程（`multiprocessing`）替代多线程。  
  - 使用C扩展（如NumPy）绕过GIL。  
  - 换用无GIL的解释器（如Jython、PyPy-STM）。  

代码示例  
```python
import threading

def cpu_bound_task():
    sum = 0
    for _ in range(107):
        sum += 1

# 多线程执行CPU密集型任务（受GIL限制）
threads = [threading.Thread(target=cpu_bound_task) for _ in range(2)]
for t in threads:
    t.start()
for t in threads:
    t.join()
# 执行时间可能接近单线程的两倍（因GIL竞争）
```



## 5. `pass`、`break` 和 `continue` 的作用是什么？
答案  
- `pass`：空语句，用于语法占位（无任何操作）。  
  - 应用：定义空函数/类，或占位未实现的代码块。  

- `break`：立即终止当前循环（跳出整个循环结构）。  
  - 应用：提前结束循环（如搜索到目标后退出）。  

- `continue`：跳过当前循环剩余代码，进入下一轮循环。  
  - 应用：忽略某些条件的数据（如过滤无效值）。  

代码示例  
```python
# pass示例
def empty_function():
    pass  # 无操作，避免语法错误

# break示例
for i in range(10):
    if i == 5:
        break  # 循环终止于i=5
    print(i)   # 输出0,1,2,3,4

# continue示例
for i in range(5):
    if i % 2 == 0:
        continue  # 跳过偶数i
    print(i)      # 输出1,3
```



## 6. 如何交换两个变量的值？
答案  
- 原理：Python的元组解包（Tuple Unpacking）特性。  
  - 右侧表达式 `b, a` 生成一个元组，左侧直接解包赋值。  
  - 无需临时变量，底层实现高效。  

代码示例  
```python
a = "Hello"
b = 100
a, b = b, a  # 交换变量
print(a)     # 100
print(b)     # "Hello"

# 扩展：交换列表元素
lst = [1, 2, 3]
lst[0], lst[2] = lst[2], lst[0]
print(lst)   # [3, 2, 1]
```



## 7. `*` 和 `` 在函数调用中的用途是什么？
答案  
- `*` 运算符：解包可迭代对象为位置参数。  
  - 适用类型：列表、元组、集合、生成器等。  
  - 应用：动态传递参数（如从列表展开）。  

- `` 运算符：解包字典为关键字参数。  
  - 字典的键必须与函数参数名匹配。  
  - 应用：传递配置参数（如 `kwargs`）。  

代码示例  
```python
def func(a, b, c=0):
    print(f"a={a}, b={b}, c={c}")

args = (1, 2)
kwargs = {"c": 3, "b": 2}

func(*args)          # a=1, b=2, c=0
func(kwargs)       # a报错（缺少a参数）
func(0, kwargs)    # a=0, b=2, c=3
```



## 8. 解释 `__name__ == "__main__"` 的作用。
答案  
- `__name__`：内置变量，表示当前模块名。  
  - 当模块被直接执行时，`__name__` 值为 `"__main__"`。  
  - 当模块被导入时，`__name__` 值为模块的文件名（不含 `.py`）。  

- 用途：  
  - 避免模块被导入时执行测试代码。  
  - 将模块同时作为脚本和库使用（如定义函数后添加测试逻辑）。  

代码示例  
```python
# my_module.py
def calculate(x):
    return x * 2

if __name__ == "__main__":
    # 直接执行时运行以下代码
    print(calculate(5))  # 输出10
    print("模块测试完成")

# 其他文件导入时：
# import my_module → 不会执行print语句
```



## 9. Python中的三元表达式如何编写？
答案  
- 语法：`value_if_true if condition else value_if_false`  
  - 注意：与C/Java的三元运算符 `condition ? a : b` 不同。  
  - 返回值取决于条件的布尔值。  

- 应用场景：简化简单的条件赋值。  
  - 复杂逻辑建议使用完整 `if-else` 结构以提高可读性。  

代码示例  
```python
# 基本用法
x = 10
result = "Even" if x % 2 == 0 else "Odd"
print(result)  # Even

# 嵌套三元表达式（可读性差，慎用）
y = 15
output = "A" if y > 20 else ("B" if y > 10 else "C")
print(output)  # B
```



## 10. 如何捕获和处理异常？写出 `try-except-finally` 的基本结构。
答案  
- `try` 块：包裹可能抛出异常的代码。  
- `except` 块：捕获并处理特定异常（可指定多个异常类型）。  
- `else` 块：当 `try` 块未发生异常时执行（可选）。  
- `finally` 块：无论是否发生异常都会执行（常用于资源清理）。  

最佳实践：  
- 按异常的具体性从高到低捕获（如先 `ValueError` 后 `Exception`）。  
- 避免裸 `except:`（会捕获所有异常，包括 `KeyboardInterrupt`）。  

代码示例  
```python
def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError as e:
        print(f"错误：除数不能为0 → {e}")
        return None
    except TypeError:
        print("错误：操作数类型不支持")
    else:
        print("计算成功")
        return result
    finally:
        print("资源清理（如关闭文件）")

print(divide(10, 2))   # 输出5.0
print(divide(10, 0))   # 输出错误信息并返回None
```
