# 错误处理与调试

## 51. 如何自定义异常类？它与内置异常有何区别？
详细答案  
- 自定义异常类：  
  - 继承自 `Exception` 类或其子类，可添加额外属性和方法。  
  - 用途：定义应用特定的错误类型，提高代码可读性和错误处理针对性。  

- 与内置异常的区别：  
  - 内置异常（如 `ValueError`）是Python标准库定义的，自定义异常用于领域特定逻辑。  

代码示例  
```python
class InvalidEmailError(Exception):
    """当邮箱格式无效时抛出"""
    def __init__(self, email):
        super().__init__(f"邮箱格式无效: {email}")
        self.email = email

def validate_email(email):
    if "@" not in email:
        raise InvalidEmailError(email)

try:
    validate_email("user.example.com")
except InvalidEmailError as e:
    print(e)  # 输出：邮箱格式无效: user.example.com
```



## 52. 解释Python中的断言（assert）及其适用场景。
详细答案  
- 断言：  
  - 语法：`assert condition, message`。若条件为 `False`，抛出 `AssertionError`。  
  - 用途：用于调试阶段验证程序内部状态的正确性（如前置条件、后置条件）。  

- 适用场景：  
  - 测试关键假设，如函数参数合法性。  
  - 注意：生产代码中应避免依赖断言（因可通过 `-O` 选项禁用）。  

代码示例  
```python
def divide(a, b):
    assert b != 0, "除数不能为0"
    return a / b

divide(10, 0)  # 触发AssertionError: 除数不能为0
```



## 53. 如何使用pdb进行Python代码调试？
详细答案  
- pdb（Python Debugger）：  
  - 启动方式：  
    1. 命令行：`python -m pdb script.py`。  
    2. 代码中插入 `import pdb; pdb.set_trace()`。  
  - 常用命令：  
    - `l(ist)`：查看代码。  
    - `n(ext)`：执行下一行。  
    - `s(tep)`：进入函数内部。  
    - `c(ontinue)`：继续执行至断点或结束。  
    - `p(rint) var`：打印变量值。  

代码示例  
```python
def calculate(a, b):
    import pdb; pdb.set_trace()  # 设置断点
    return a + b * 2

calculate(1, 2)
```



## 54. 解释 `__debug__` 变量的作用。
详细答案  
- `__debug__`：  
  - 内置布尔常量，默认值为 `True`。当使用 `-O` 选项运行Python时，值为 `False`。  
  - 用途：结合断言使用，优化调试代码。  

代码示例  
```python
if __debug__:
    print("调试模式已启用")
else:
    print("优化模式（-O选项）")
```



## 55. 如何记录Python程序的日志？`logging`模块的基本配置。
详细答案  
- `logging`模块：  
  - 日志级别：`DEBUG` < `INFO` < `WARNING` < `ERROR` < `CRITICAL`。  
  - 基本配置：`logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')`。  

代码示例  
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

logging.info("程序启动")
try:
    1 / 0
except Exception as e:
    logging.error("发生异常: %s", e)
```



## 56. 解释Python中的警告（Warnings）与异常的区别，如何控制警告信息？
详细答案  
- 警告（Warnings）：  
  - 用于提示不推荐用法或潜在问题，但不会终止程序（如 `DeprecationWarning`）。  
  - 异常（Exceptions）：程序错误，必须处理否则终止。  

- 控制警告：  
  - `warnings.filterwarnings("ignore")`：忽略特定警告。  
  - `-W ignore` 命令行选项：全局忽略警告。  

代码示例  
```python
import warnings

def deprecated_func():
    warnings.warn("此函数已弃用", DeprecationWarning)
    return 42

# 过滤警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
deprecated_func()  # 无警告输出
```



## 57. 在Python中如何实现性能分析（Profiling）？
详细答案  
- 性能分析工具：  
  - cProfile：统计函数调用时间和次数。  
  - line_profiler：逐行分析代码耗时。  
  - memory_profiler：分析内存使用。  

代码示例  
```python
# cProfile示例
import cProfile

def slow_function():
    sum(range(106))

cProfile.run("slow_function()", sort="cumulative")

# 输出结果：
# 1000003 function calls in 0.025 seconds
# Ordered by: cumulative time
# ...
```



## 58. 如何优化Python代码的性能？常见的优化技巧有哪些？
详细答案  
- 优化技巧：  
  1. 使用内置函数和库：如用 `map` 替代循环。  
  2. 避免全局变量：局部变量访问更快。  
  3. 使用数据结构优化：如集合（`set`）的 `O(1)` 查找。  
  4. JIT编译：使用PyPy或Numba加速计算密集型代码。  
  5. 减少函数调用开销：避免在循环中频繁调用函数。  

代码示例  
```python
# 优化前（慢）
result = []
for i in range(10000):
    result.append(i * 2)

# 优化后（快）
result = [i * 2 for i in range(10000)]
```



## 59. 如何捕获和处理多个异常？
详细答案  
- 方法：  
  - 在 `except` 子句中指定多个异常类型（使用元组）。  
  - 使用多个 `except` 块分别处理不同异常。  

代码示例  
```python
try:
    # 可能抛出 ValueError 或 TypeError
    x = int("NaN")
except (ValueError, TypeError) as e:
    print(f"输入错误: {e}")
except Exception as e:
    print(f"未知异常: {e}")
```



## 60. 解释Python中的`traceback`模块及其用途。
详细答案  
- `traceback`模块：  
  - 用于提取和格式化异常的回溯信息。  
  - 核心函数：  
    - `traceback.format_exc()`：返回异常信息的字符串。  
    - `traceback.print_exc()`：直接打印异常信息。  

代码示例  
```python
import traceback

try:
    1 / 0
except ZeroDivisionError:
    with open("error.log", "w") as f:
        f.write(traceback.format_exc())
    traceback.print_exc()  # 打印到控制台
```